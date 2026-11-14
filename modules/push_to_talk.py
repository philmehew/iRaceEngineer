"""
Push-to-talk controller — wheel-button hold-to-talk voice input.

Extracted from live_mode's nested closures. Holds the recording/generation
state (recording event, monotonic start time, stuck-thread force-clear
generation counter) and spawns the record → transcribe → query thread.

Decoupled from the LLM stack via an `on_question(question, timing_steps)`
callback, so live_mode decides what to do with the transcript.
"""

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("iraceengineer")


class PushToTalkController:
    """Coordinate wheel-button push-to-talk recording and transcription.

    on_press starts a recording (spawning a background record→transcribe
    thread); on_release signals the recording to stop. A 30s stuck-recording
    guard force-clears a hung thread via a generation counter so its finally
    block can't clobber a newer recording's state.

    Callers hand controller.on_press / controller.on_release to a
    WheelButtonListener, and construct with an on_question callback that
    receives the transcript (plus timing steps) when speech is detected.
    """

    def __init__(
        self,
        stt,
        *,
        on_question: Callable[[str, list[tuple[str, float]]], None],
        max_record_seconds: float = 15,
    ):
        self._stt = stt
        self._on_question = on_question
        self._max_record_seconds = max_record_seconds

        self._recording = threading.Event()
        self._stop = threading.Event()
        self._recording_time = [0.0]  # monotonic time when recording started
        self._generation = [0]  # increments per recording; lets a stale thread
        # detect it was force-cleared and avoid clobbering a newer recording's
        # flags in its finally block

    def on_press(self):
        """Start recording when the wheel button is pressed."""
        if self._recording.is_set():
            # Previous recording still in progress — could be a stuck thread.
            # If it's been running >30s, it's stuck; force-clear.
            elapsed = time.monotonic() - self._recording_time[0]
            if elapsed > 30:
                logger.warning(
                    f"PTT stuck for {elapsed:.0f}s — force-clearing "
                    f"(previous recording thread likely hung)"
                )
                # Retire the old thread: bump the generation so its finally
                # block won't touch the new recording's state
                self._generation[0] += 1
                self._stop.set()
                self._recording.clear()
            else:
                logger.debug(f"PTT still recording ({elapsed:.1f}s) — ignoring press")
                return
        generation = self._generation[0]
        self._recording.set()
        self._recording_time[0] = time.monotonic()
        self._stop.clear()

        def _record_and_query():
            """Record, transcribe, and query LLM with timing."""
            try:
                timing_steps = []
                logger.info("Voice button pressed — recording...")
                t0 = time.monotonic()
                audio = self._stt.record_until_release(
                    self._stop,
                    max_duration_s=self._max_record_seconds,
                )
                t1 = time.monotonic()
                timing_steps.append(("Recording", t1 - t0))

                text = self._stt.transcribe(audio)
                t2 = time.monotonic()
                timing_steps.append(("Transcription", t2 - t1))

                if text.strip():
                    logger.info(f"Transcribed: {text}")
                    self._on_question(text, timing_steps)
                else:
                    logger.warning("No speech detected — skipping LLM query")
                    from modules.engineer_query import _print_timing_summary

                    _print_timing_summary(timing_steps, label="Voice (no speech)")
            except Exception:
                logger.exception("Voice recording/transcription failed")
            finally:
                # Only clear flags if this thread still owns the current
                # recording — a force-cleared thread must not clobber a newer
                # recording's state
                if self._generation[0] == generation:
                    self._recording.clear()

        threading.Thread(target=_record_and_query, daemon=True).start()

    def on_release(self):
        """Stop recording when the wheel button is released."""
        self._stop.set()
