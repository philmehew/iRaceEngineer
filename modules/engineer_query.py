"""
Race engineer query pipeline — build context, call the LLM, process the reply.

Homes the module-level locks that protect live mode's state reads/writes
between the main poll loop and the wheel-button-hook thread, and prevents
double-press:

- `_state_lock` — serializes state reads (snapshot for the prompt) against the
  poll loop's state.update(). Imported by live_mode / replay_mode so every
  caller shares the same lock.
- `_llm_call_lock` — non-blocking guard on LLM calls: acquire(blocking=False)
  is atomic, unlike an Event's check-then-set, so two threads can never both
  start a call. handle_button_press uses it to skip a double-press.

`handle_button_press()` is the live-mode query entry point (called by the
wheel listener and CLI-question thread). `ask_engineer()` is the shared seam
that also serves replay mode, killing the duplicated inline query blocks.
"""

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from modules.chat_macros import ChatMacroManager
from modules.context_builder import ContextBuilder
from modules.llm_client import LLMClient, LLMError
from modules.logging_setup import llm_query_logger
from modules.race_state import RaceState
from modules.tts_client import TTSClient

if TYPE_CHECKING:
    from modules.iracing_client import IRacingClient

logger = logging.getLogger("iraceengineer")

# Thread safety for live mode: protects state reads/writes between the main
# poll loop and the wheel-button-hook thread, and prevents double-press.
_state_lock = threading.Lock()
# Non-blocking lock guarding LLM calls: acquire(blocking=False) is atomic,
# unlike an Event's check-then-set, so two threads can never both start a call.
_llm_call_lock = threading.Lock()


def _print_timing_summary(steps: list[tuple[str, float]], label: str = "Pipeline"):
    """Print a timing breakdown for the voice/LLM pipeline.

    Args:
        steps: List of (step_name, duration_seconds) tuples.
        label: Header label for the summary block.
    """
    if not steps:
        return

    total = sum(duration for _, duration in steps)
    print(f"\n⏱  {label} timing:")
    for name, duration in steps:
        print(f"  {name:25s} {duration:.3f}s")
    print(f"  {'─' * 35}")
    print(f"  {'Total':25s} {total:.3f}s\n")


def _clean_response(text: str) -> str:
    """Post-process LLM response text for common model quirks."""
    # Replace tildes with "about" (small models often output ~2 instead of "about 2")
    text = text.replace("~", "about ")
    # Expand unit abbreviations TTS engines read poorly ("\m slash s")
    text = re.sub(r"\bm/s\b", "meters per second", text)
    # kph (or km/h) — TTS voices read these as letter soup
    text = re.sub(r"\bkm/h\b", "kilometers per hour", text)
    text = re.sub(r"\bkph\b", "kilometers per hour", text)
    # PSI: TTS voices say "sigh" or "psye"; force letter-by-letter "P S I"
    text = re.sub(r"\bPSI\b", "P S I", text)
    # Standalone L after a number = litres ("12.5 L" -> "12.5 litres")
    text = re.sub(r"(\d)\s*L\b", r"\1 litres", text)
    text = re.sub(r"\b1 litres\b", "1 litre", text)
    return text


def ask_engineer(
    state: RaceState,
    context_builder: ContextBuilder,
    llm: LLMClient,
    question: str,
    *,
    chat_macros: ChatMacroManager | None = None,
    log_label: str = "",
    iracing_client=None,
) -> str | None:
    """Shared LLM query seam used by live and replay modes.

    Builds the prompt from a locked state snapshot, calls the LLM, logs the
    exchange to the query log, cleans the response, fires [CMD:] chat macro
    actions, and appends the exchange to QA history.

    Callers own display formatting, TTS, and timing summaries, which genuinely
    differ per mode. Returns the cleaned response text, or None on error/empty.

    Args:
        chat_macros: Optional ChatMacroManager — when present, [CMD:name] tags
            in the LLM response are stripped and the matching chat macro fired.
        log_label: Suffix for the query-log header ("", "(replay)", ...).
        iracing_client: Client used to send chat macros. None = log-only
            (replay mode — no live sim to send through).
    """
    # Build the prompt under the state lock for a consistent snapshot
    with _state_lock:
        snapshot = state.get_snapshot()
    messages = context_builder.build_prompt(snapshot, question=question)

    depth = context_builder.context_depth
    logger.info(
        f"Sending context (depth={depth}, ~{len(str(messages[-1]['content']))} bytes) to LLM..."
    )

    try:
        response_text = llm.ask(messages)
    except LLMError as e:
        # LLM failure — show a short user-facing message, don't speak
        # raw error text as if it were engineer advice
        print(f"\n⚠️  {e}")
        logger.warning(f"LLM call failed: {e}")
        return None

    # Log the prompt and response to the LLM query log file
    log_suffix = f" {log_label}" if log_label else ""
    llm_query_logger.info(
        f"--- LLM QUERY{log_suffix} ---\n"
        f"Depth: {depth}\n"
        f"Question: {question or '(none)'}\n\n"
        "=== PROMPT SENT ===\n"
        f"{json.dumps(messages, indent=2, ensure_ascii=False)}\n\n"
        "=== RESPONSE ===\n"
        f"{response_text or '(empty)'}\n"
        "--- END ---"
    )

    if not response_text:
        logger.warning("LLM returned empty response")
        return None

    # Post-process: clean common model quirks
    response_text = _clean_response(response_text)

    # Extract [CMD:name] action tags before display/TTS so they're never
    # spoken, then fire the corresponding chat macros
    if chat_macros is not None and chat_macros.enabled:
        response_text, action_names = chat_macros.parse_response(response_text)
        for action_name in action_names:
            chat_macros.fire(action_name, iracing_client=iracing_client)

    # Record the exchange so follow-up questions have context
    context_builder.qa_history.append(question or "(strategy update)", response_text)

    return response_text


def handle_button_press(
    state: RaceState,
    context_builder: ContextBuilder,
    llm: LLMClient,
    question: str = "",
    tts: TTSClient | None = None,
    timing_steps: list[tuple[str, float]] | None = None,
    chat_macros: ChatMacroManager | None = None,
    iracing: "IRacingClient | None" = None,
):
    """Handle a button press — build context, call LLM, process response.

    Thread-safe: acquires the state lock to read a consistent snapshot,
    and guards against double-press (skips if an LLM call is already in progress).

    Args:
        timing_steps: Optional list of (step_name, duration) tuples from
            earlier pipeline steps (e.g. recording, transcription). Additional
            timing will be appended and a summary printed at the end.
        chat_macros: Optional ChatMacroManager — when present, [CMD:name] tags
            in the LLM response are stripped and the matching chat macro fired.
        iracing: Client used to send chat macros (None = log-only, for tests).
    """
    # Guard against double-press — skip if an LLM call is already running.
    # acquire(blocking=False) is atomic, so two racing threads can't both pass.
    if not _llm_call_lock.acquire(blocking=False):
        logger.info("LLM call already in progress — skipping")
        return

    steps = list(timing_steps) if timing_steps else []

    try:
        logger.info("Button pressed — querying LLM...")

        # Get a consistent snapshot under the state lock
        t0 = time.monotonic()
        with _state_lock:
            snapshot = state.get_snapshot()
        steps.append(("State snapshot", time.monotonic() - t0))

        # Build context
        t0 = time.monotonic()
        messages = context_builder.build_prompt(snapshot, question=question)
        steps.append(("Context build", time.monotonic() - t0))

        # Log the context being sent
        depth = context_builder.context_depth
        logger.info(
            f"Sending context (depth={depth}, ~{len(str(messages[-1]['content']))} bytes) to LLM..."
        )

        # Call LLM
        t0 = time.monotonic()
        try:
            response_text = llm.ask(messages)
        except LLMError as e:
            # LLM failure — show a short user-facing message, don't speak
            # raw error text as if it were engineer advice
            steps.append(("LLM call", time.monotonic() - t0))
            print(f"\n⚠️  {e}")
            logger.warning(f"LLM call failed: {e}")
            _print_timing_summary(steps, label="Pipeline (error)")
            return
        steps.append(("LLM call", time.monotonic() - t0))

        # Log the prompt and response to the LLM query log file
        llm_query_logger.info(
            "--- LLM QUERY ---\n"
            f"Depth: {depth}\n"
            f"Question: {question or '(none)'}\n\n"
            "=== PROMPT SENT ===\n"
            f"{json.dumps(messages, indent=2, ensure_ascii=False)}\n\n"
            "=== RESPONSE ===\n"
            f"{response_text or '(empty)'}\n"
            "--- END ---"
        )

        if not response_text:
            logger.warning("LLM returned empty response")
            _print_timing_summary(steps)
            return

        # Post-process: clean common model quirks
        response_text = _clean_response(response_text)

        # Extract [CMD:name] action tags before display/TTS so they're never
        # spoken, then fire the corresponding chat macros
        fired_actions: list[str] = []
        if chat_macros is not None and chat_macros.enabled:
            response_text, action_names = chat_macros.parse_response(response_text)
            for action_name in action_names:
                if chat_macros.fire(action_name, iracing_client=iracing):
                    fired_actions.append(action_name)

        # Record the exchange so follow-up questions have context
        context_builder.qa_history.append(
            question or "(strategy update)", response_text
        )

        # Display response
        print("\n" + "=" * 60)
        print(f"🏁 RACE ENGINEER (depth={depth})")
        print("=" * 60)
        print(response_text)
        print("=" * 60)

        # Speak the response if TTS is enabled
        if tts is not None:
            tts.speak_async(response_text)

        # Print timing summary
        _print_timing_summary(steps)

    finally:
        _llm_call_lock.release()
