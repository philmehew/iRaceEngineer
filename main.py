"""
iRaceEngineer — Real-time iRacing data collection with on-demand LLM strategy.

Collects telemetry from iRacing via shared memory (pyirsdk), maintains
an in-memory race state, and sends a condensed snapshot to an OpenAI-compatible
LLM when a button is pressed.

Usage:
    python main.py                    # Normal mode — connect to iRacing
    python main.py --capture          # Capture mode — record telemetry to JSON
    python main.py --replay <dir>     # Replay mode — feed captured data through pipeline
    python main.py --generate-samples # Generate sample data for testing
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import yaml
from datetime import datetime

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from iracing_client import IRacingClient
from race_state import RaceState, TyreState
from context_builder import ContextBuilder
from llm_client import LLMClient
from action_executor import ActionExecutor
from capture import TelemetryCapture, TelemetryReplay, create_sample_data
from stt_client import STTClient
from tts_client import TTSClient
from spotter import Spotter


def _ensure_cuda_dlls() -> bool:
    """Register CUDA toolkit DLL directory on Windows.

    The CUDA installer sometimes doesn't add itself to PATH, causing
    "DLL not found" errors from ctranslate2/faster-whisper. This scans
    for the toolkit and registers the bin directory so the DLLs are findable.

    Must be called before any code that loads CUDA (e.g. stt_client preload).

    Returns:
        True if CUDA DLL directory was found and registered, False otherwise.
    """
    if sys.platform != "win32":
        return False

    # Try CUDA_PATH env var first (set by some installer versions)
    cuda_base = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
    if os.path.isdir(cuda_base):
        os.add_dll_directory(cuda_base)
        return True

    # Fallback: scan the standard toolkit install location
    toolkit_dir = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        "NVIDIA GPU Computing Toolkit",
        "CUDA",
    )
    if not os.path.isdir(toolkit_dir):
        return False

    # Pick the latest installed version
    versions = sorted(
        (d for d in os.listdir(toolkit_dir) if d.startswith("v")),
        reverse=True,
    )
    if not versions:
        return False

    cuda_bin = os.path.join(toolkit_dir, versions[0], "bin")
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
        return True

    return False


class WheelButtonListener:
    """Listen for joystick/wheel button presses using pygame.

    Runs a background thread that polls pygame events and calls
    on_press/on_release callbacks when the target button is triggered.

    This is used for push-to-talk on a steering wheel — hold the button
    to record, release to transcribe.

    Config keys (under voice.trigger):
        device_index: Joystick device index (use test_wheel.py to find)
        button_index: Button index on the device
    """

    def __init__(
        self,
        device_index: int,
        button_index: int,
        on_press=None,
        on_release=None,
    ):
        self.device_index = device_index
        self.button_index = button_index
        self.on_press = on_press
        self.on_release = on_release
        self._running = False
        self._thread: threading.Thread | None = None
        self._joystick = None
        self._pygame_initialized = False

    def start(self):
        """Start listening for button events in a background thread."""
        if self._running:
            return

        try:
            import importlib.util

            if not importlib.util.find_spec("pygame"):
                raise ImportError
        except (ImportError, ModuleNotFoundError):
            logger.error("pygame not installed. Install with: uv sync --extra wheel")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and clean up pygame."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._pygame_initialized:
            try:
                import pygame

                if self._joystick is not None:
                    self._joystick.quit()
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False

    def _run(self):
        """Background thread: init pygame, open joystick, poll button state.

        Uses direct get_button() polling to detect press/release edges.
        This approach is more reliable than pygame's event system because:
        - It works in background threads on Windows
        - It doesn't require a display window
        - It doesn't depend on the Windows message pump

        Falls back to event-based detection if polling reports stale state
        (all buttons permanently pressed), which can happen if the event pump
        isn't running.
        """
        try:
            import pygame
        except ImportError:
            logger.error("pygame not installed — cannot use wheel button trigger")
            self._running = False
            return

        pygame.init()
        pygame.joystick.init()
        self._pygame_initialized = True

        count = pygame.joystick.get_count()
        logger.info(f"Wheel button listener: pygame init, {count} joystick(s) found")

        if self.device_index >= count:
            logger.error(
                f"Joystick device index {self.device_index} not found "
                f"(only {count} device(s) available). "
                f"Run test_wheel.py --list to see devices."
            )
            self._running = False
            pygame.quit()
            self._pygame_initialized = False
            return

        self._joystick = pygame.joystick.Joystick(self.device_index)
        self._joystick.init()
        js_name = self._joystick.get_name()
        js_buttons = self._joystick.get_numbuttons()
        logger.info(
            f"Wheel button listener: device {self.device_index} "
            f"({js_name}), button {self.button_index} of {js_buttons}"
        )

        # Track button state to detect press/release edges
        was_pressed = False
        last_log_time = 0

        try:
            while self._running:
                # Pump pygame events so joystick state stays fresh.
                # Even though we read state via get_button(), the internal
                # joystick state only updates when the event pump runs.
                pygame.event.pump()

                try:
                    is_pressed = bool(self._joystick.get_button(self.button_index))
                except Exception:
                    # Joystick may have disconnected — try to reconnect next tick
                    time.sleep(1 / 30)
                    continue

                if is_pressed and not was_pressed:
                    was_pressed = True
                    logger.info(
                        f"Wheel button {self.button_index} pressed "
                        f"(device {self.device_index}: {js_name})"
                    )
                    if self.on_press:
                        self.on_press()
                elif not is_pressed and was_pressed:
                    was_pressed = False
                    logger.info(
                        f"Wheel button {self.button_index} released "
                        f"(device {self.device_index}: {js_name})"
                    )
                    if self.on_release:
                        self.on_release()

                # Periodic alive log (every 30s) to confirm thread is running
                now = time.monotonic()
                if now - last_log_time > 30:
                    last_log_time = now
                    logger.debug(
                        f"Wheel button listener alive: device {self.device_index}, "
                        f"button {self.button_index}, state={'pressed' if was_pressed else 'released'}"
                    )

                time.sleep(1 / 30)  # ~30Hz polling

        except Exception as e:
            logger.error(f"Wheel button listener error: {e}")
        finally:
            if self._joystick is not None:
                self._joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
            self._pygame_initialized = False


def _session_timestamp() -> str:
    """Return a timestamp string for log file names (same format as capture sessions)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _insert_timestamp(filename: str, ts: str) -> str:
    """Insert a timestamp before the .log extension in a filename.

    e.g. _insert_timestamp("llm_queries.log", "2024-06-08_14-30-00")
         -> "llm_queries_2024-06-08_14-30-00.log"
    """
    if filename.endswith(".log"):
        return f"{filename[:-4]}_{ts}.log"
    return f"{filename}_{ts}"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iraceengineer")

# Dedicated logger for LLM query data — handler added in setup_llm_query_log()
llm_query_logger = logging.getLogger("iraceengineer.llm_query")
llm_query_logger.propagate = False  # Don't double-print to console

# Thread safety for live mode: protects state reads/writes between the main
# poll loop and the keyboard-hook thread, and prevents double-press.
_state_lock = threading.Lock()
_llm_in_progress = threading.Event()


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


def setup_llm_query_log(config: dict, session_ts: str):
    """Set up the LLM query log file handler.

    Creates a FileHandler that writes the prompt and response for every LLM
    call to a timestamped log file (e.g. logs/llm_queries_2024-06-08_14-30-00.log).
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("llm_query_log_dir", "logs")
    log_file = log_config.get("llm_query_log_file", "llm_queries.log")

    os.makedirs(log_dir, exist_ok=True)
    stamped_file = _insert_timestamp(log_file, session_ts)
    log_path = os.path.join(log_dir, stamped_file)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s\n%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    llm_query_logger.addHandler(handler)
    llm_query_logger.setLevel(logging.INFO)
    logger.info(f"LLM query log: {log_path}")


def setup_console_log(config: dict, session_ts: str):
    """Set up a console mirror log that captures all console output to a file.

    Adds a FileHandler to the root logger so anything printed to the console
    is also written to a timestamped log file (e.g. logs/console_2024-06-08_14-30-00.log).
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("console_log_dir", "logs")
    log_file = log_config.get("console_log_file", "console.log")

    os.makedirs(log_dir, exist_ok=True)
    stamped_file = _insert_timestamp(log_file, session_ts)
    log_path = os.path.join(log_dir, stamped_file)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    # No explicit level — inherits from root logger, so mirrors whatever the console shows
    logging.getLogger().addHandler(handler)
    logger.info(f"Console log: {log_path}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config or {}


def handle_button_press(
    state: RaceState,
    context_builder: ContextBuilder,
    llm: LLMClient,
    executor: ActionExecutor,
    question: str = "",
    tts: TTSClient | None = None,
    timing_steps: list[tuple[str, float]] | None = None,
):
    """Handle a button press — build context, call LLM, process response.

    Thread-safe: acquires the state lock to read a consistent snapshot,
    and guards against double-press (skips if an LLM call is already in progress).

    Args:
        timing_steps: Optional list of (step_name, duration) tuples from
            earlier pipeline steps (e.g. recording, transcription). Additional
            timing will be appended and a summary printed at the end.
    """
    # Guard against double-press — skip if an LLM call is already running
    if _llm_in_progress.is_set():
        logger.info("LLM call already in progress — skipping")
        return
    _llm_in_progress.set()

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
        response_text = llm.ask(messages)
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

        # Parse actions from response
        t0 = time.monotonic()
        clean_text, actions = executor.parse_response(response_text)
        steps.append(("Action parse", time.monotonic() - t0))

        # Display response
        print("\n" + "=" * 60)
        print(f"🏁 RACE ENGINEER (depth={depth})")
        print("=" * 60)
        print(clean_text)
        if actions:
            print("\n--- Actions ---")
            results = executor.execute(actions)
            for result in results:
                print(f"  {result}")
        print("=" * 60)

        # Speak the response if TTS is enabled
        if tts is not None:
            tts.speak_async(clean_text)

        # Print timing summary
        _print_timing_summary(steps)

    finally:
        _llm_in_progress.clear()


def run_live_mode(config: dict, tick_rate_hz: int = 30):
    """Main loop — connect to iRacing and poll telemetry."""
    iracing = IRacingClient()
    state = RaceState(config)
    context_builder = ContextBuilder(config)
    llm = LLMClient(config)
    executor = ActionExecutor(iracing, config)

    # Set up voice clients (graceful if not installed)
    voice_config = config.get("voice", {})
    tts = None
    if voice_config.get("tts", {}).get("enabled", False):
        try:
            tts = TTSClient(config)
            if tts.is_available:
                logger.info("TTS enabled — LLM responses will be spoken aloud")
            else:
                logger.warning("TTS configured but piper-tts not installed — disabled")
                tts = None
        except Exception as e:
            logger.warning(f"TTS setup failed: {e}")
            tts = None

    stt = None
    if voice_config.get("stt", {}).get("enabled", False):
        try:
            stt = STTClient(config)
            if stt.is_available:
                logger.info("STT enabled — voice input available via push-to-talk")
            else:
                logger.warning(
                    "STT configured but faster-whisper not installed — disabled"
                )
                stt = None
        except Exception as e:
            logger.warning(f"STT setup failed: {e}")
            stt = None

    # Set up spotter (real-time audio calls for car proximity)
    spotter = None
    spotter_config = config.get("spotter", {})
    if spotter_config.get("enabled", False):
        try:
            spotter = Spotter(config)
            if spotter.enabled:
                logger.info("Spotter enabled — car proximity calls active")
            else:
                spotter = None
        except Exception as e:
            logger.warning(f"Spotter setup failed: {e}")

    # Ensure CUDA DLLs are registered before loading voice models
    cuda_dll_found = _ensure_cuda_dlls()

    # Pre-load voice models to avoid cold-start latency on first PTT press
    if stt is not None:
        if stt.device == "cuda" and not cuda_dll_found:
            logger.warning(
                "CUDA toolkit DLLs not found — STT will fall back to CPU. "
                "Install CUDA or add it to PATH for GPU acceleration."
            )
        stt.preload()
    if tts is not None:
        tts.preload()

    # Set up keyboard trigger
    trigger_key = config.get("trigger", {}).get("key", "f9")
    voice_trigger_config = voice_config.get("trigger", {})
    voice_trigger_method = voice_trigger_config.get("method", "keyboard")
    voice_key = voice_trigger_config.get("voice_key", "f10")
    try:
        import keyboard  # noqa: F401 — testing availability

        keyboard_available = True
        logger.info(f"Trigger key: {trigger_key.upper()} (press to query LLM)")
    except ImportError:
        keyboard_available = False
        logger.warning("keyboard module not available — using Enter key for trigger")

    # Connect to iRacing
    print("\n🔌 Connecting to iRacing...")
    if not iracing.startup():
        print(
            "❌ iRacing not found. Make sure iRacing is running with an active session."
        )
        print("   Retrying every 5 seconds... Press Ctrl+C to exit.")

        while not iracing.startup():
            time.sleep(5)
            if iracing.is_connected:
                break

    print("✅ Connected to iRacing!\n")

    # Detect teammates
    team_indices = iracing.detect_team_indices(config)
    driver_aliases = getattr(iracing, "driver_aliases", {})
    state.set_team_indices(team_indices, driver_aliases)
    driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
    state.set_driver_names(driver_names)
    print(
        f"👥 Team detected: {len(team_indices)} cars — {', '.join(driver_names.get(i, f'Car #{i}') for i in team_indices)}\n"
    )

    # Register triggers (keyboard for LLM query, keyboard or wheel for voice PTT)
    wheel_listener = None

    if keyboard_available:
        import keyboard as kb

        # F9 (or configured key) — LLM query trigger (always keyboard)
        kb.add_hotkey(
            trigger_key,
            lambda: handle_button_press(state, context_builder, llm, executor, tts=tts),
        )
        print(f"🎧 Listening for {trigger_key.upper()} key press...")
        print(f"   Press {trigger_key.upper()} to ask the race engineer a question.")

        # Voice PTT — keyboard method
        if stt is not None and voice_trigger_method == "keyboard":
            # Push-to-talk: record while key is held, transcribe on release
            _ptt_recording = threading.Event()
            _ptt_recording_time = [0.0]  # monotonic time when recording started
            _ptt_stop = threading.Event()

            def _on_voice_key_down():
                """Start recording when voice key is pressed."""
                if _ptt_recording.is_set():
                    # Previous recording still in progress — could be a stuck
                    # thread. If it's been running >30s, it's stuck; force-clear.
                    elapsed = time.monotonic() - _ptt_recording_time[0]
                    if elapsed > 30:
                        logger.warning(
                            f"PTT stuck for {elapsed:.0f}s — force-clearing "
                            f"(previous recording thread likely hung)"
                        )
                        _ptt_recording.clear()
                        _ptt_stop.set()
                    else:
                        logger.debug(
                            f"PTT still recording ({elapsed:.1f}s) — ignoring press"
                        )
                        return
                _ptt_recording.set()
                _ptt_recording_time[0] = time.monotonic()
                _ptt_stop.clear()

                def _record_and_query():
                    """Record, transcribe, and query LLM with timing."""
                    try:
                        timing_steps = []
                        logger.info("Voice key pressed — recording...")
                        t0 = time.monotonic()
                        audio = stt.record_until_release(
                            _ptt_stop,
                            max_duration_s=voice_trigger_config.get(
                                "max_record_seconds", 15
                            ),
                        )
                        t1 = time.monotonic()
                        timing_steps.append(("Recording", t1 - t0))

                        text = stt.transcribe(audio)
                        t2 = time.monotonic()
                        timing_steps.append(("Transcription", t2 - t1))

                        if text.strip():
                            logger.info(f"Transcribed: {text}")
                            handle_button_press(
                                state,
                                context_builder,
                                llm,
                                executor,
                                question=text,
                                tts=tts,
                                timing_steps=timing_steps,
                            )
                        else:
                            logger.warning("No speech detected — skipping LLM query")
                            _print_timing_summary(
                                timing_steps, label="Voice (no speech)"
                            )
                    except Exception:
                        logger.exception("Voice recording/transcription failed")
                    finally:
                        _ptt_recording.clear()

                threading.Thread(target=_record_and_query, daemon=True).start()

            def _on_voice_key_up():
                """Stop recording when voice key is released."""
                _ptt_stop.set()

            # Register press/release hooks for push-to-talk
            kb.on_press_key(voice_key, lambda e: _on_voice_key_down())
            kb.on_release_key(voice_key, lambda e: _on_voice_key_up())
            print(f"   Hold {voice_key.upper()} to speak your question (push-to-talk).")

    # Voice PTT — wheel button method
    if stt is not None and voice_trigger_method == "wheel_button":
        device_index = voice_trigger_config.get("device_index")
        button_index = voice_trigger_config.get("button_index")
        if device_index is None or button_index is None:
            logger.error(
                "Wheel button trigger configured but device_index or button_index "
                "not set. Run test_wheel.py to discover your wheel button."
            )
            print(
                "   ⚠️  Wheel button trigger misconfigured — missing device_index/button_index."
            )
            print("   Run: python test_wheel.py to discover button indices.")
        else:
            # Push-to-talk: record while button is held, transcribe on release
            _ptt_recording = threading.Event()
            _ptt_recording_time = [0.0]  # monotonic time when recording started
            _ptt_stop = threading.Event()

            def _on_voice_button_down():
                """Start recording when wheel button is pressed."""
                if _ptt_recording.is_set():
                    # Previous recording still in progress — could be a stuck
                    # thread. If it's been running >30s, it's stuck; force-clear.
                    elapsed = time.monotonic() - _ptt_recording_time[0]
                    if elapsed > 30:
                        logger.warning(
                            f"PTT stuck for {elapsed:.0f}s — force-clearing "
                            f"(previous recording thread likely hung)"
                        )
                        _ptt_recording.clear()
                        _ptt_stop.set()
                    else:
                        logger.debug(
                            f"PTT still recording ({elapsed:.1f}s) — ignoring press"
                        )
                        return
                _ptt_recording.set()
                _ptt_recording_time[0] = time.monotonic()
                _ptt_stop.clear()

                def _record_and_query():
                    """Record, transcribe, and query LLM with timing."""
                    try:
                        timing_steps = []
                        logger.info("Wheel button pressed — recording...")
                        t0 = time.monotonic()
                        audio = stt.record_until_release(
                            _ptt_stop,
                            max_duration_s=voice_trigger_config.get(
                                "max_record_seconds", 15
                            ),
                        )
                        t1 = time.monotonic()
                        timing_steps.append(("Recording", t1 - t0))

                        text = stt.transcribe(audio)
                        t2 = time.monotonic()
                        timing_steps.append(("Transcription", t2 - t1))

                        if text.strip():
                            logger.info(f"Transcribed: {text}")
                            handle_button_press(
                                state,
                                context_builder,
                                llm,
                                executor,
                                question=text,
                                tts=tts,
                                timing_steps=timing_steps,
                            )
                        else:
                            logger.warning("No speech detected — skipping LLM query")
                            _print_timing_summary(
                                timing_steps, label="Voice (no speech)"
                            )
                    except Exception:
                        logger.exception("Voice recording/transcription failed")
                    finally:
                        _ptt_recording.clear()

                threading.Thread(target=_record_and_query, daemon=True).start()

            def _on_voice_button_up():
                """Stop recording when wheel button is released."""
                _ptt_stop.set()

            wheel_listener = WheelButtonListener(
                device_index=device_index,
                button_index=button_index,
                on_press=_on_voice_button_down,
                on_release=_on_voice_button_up,
            )
            wheel_listener.start()
            print(
                f"   Hold wheel button {button_index} (device {device_index}) "
                f"to speak your question (push-to-talk)."
            )

    print("   Press Ctrl+C to exit.\n")
    tick_interval = 1.0 / tick_rate_hz
    try:
        while True:
            if iracing.is_connected:
                # Read telemetry
                telemetry = iracing.get_telemetry()
                session_info = iracing.get_session_info()

                # Read telemetry units for unit conversion (kPa, degF, etc.)
                # Only needed once per session, but safe to call each tick
                try:
                    units = iracing.get_telemetry_units()
                except Exception:
                    units = {}

                # Update race state (under lock to prevent stale reads from F9 handler)
                driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
                with _state_lock:
                    state.update(telemetry, session_info, driver_names, units=units)

                # Spotter tick — car proximity audio calls + fuel/flag/pit alerts
                if spotter is not None:
                    car_lr = state.player.car_left_right
                    on_track = state.player.is_on_track
                    track_surface = state.player.player_track_surface
                    fuel_laps = state.fuel_laps_remaining
                    flags = state.session.flags
                    session_state = state.session.session_state
                    incidents = state.player.incidents
                    on_pit_road = state.player.on_pit_road

                    # Find car directly behind player for closing-approach alert
                    car_behind_gap = 0.0
                    car_behind_lap_time = -1.0
                    for car in state.nearby_cars:
                        if car.gap_seconds < 0:  # negative = behind player
                            # Closest behind = least negative gap
                            if (
                                car_behind_gap == 0.0
                                or car.gap_seconds > car_behind_gap
                            ):
                                car_behind_gap = car.gap_seconds
                                car_behind_lap_time = car.last_lap_time

                    spotter.update(
                        car_lr,
                        is_on_track=on_track,
                        track_surface=track_surface,
                        fuel_laps_remaining=fuel_laps,
                        session_flags=flags,
                        session_state=session_state,
                        incidents=incidents,
                        on_pit_road=on_pit_road,
                        car_behind_gap=car_behind_gap,
                        car_behind_lap_time=car_behind_lap_time,
                        player_last_lap_time=state.player.last_lap_time,
                        lap_completed=state.player.lap_completed,
                    )

            else:
                # iRacing disconnected — try to reconnect
                logger.warning("iRacing disconnected, attempting reconnect...")
                if not iracing.startup():
                    time.sleep(2)
                    continue
                print("✅ Reconnected to iRacing")
                # Reset spotter state on reconnect to avoid stale transitions
                if spotter is not None:
                    spotter.reset()

            time.sleep(tick_interval)

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if wheel_listener is not None:
            wheel_listener.stop()
        if keyboard_available:
            kb.unhook_all()
        iracing.shutdown()


def run_capture_mode(config: dict, capture_dir: str, interval_ms: int = 1000):
    """Capture mode — record telemetry snapshots to JSON files."""
    iracing = IRacingClient()
    state = RaceState(config)
    capture = TelemetryCapture(capture_dir, interval_ms)

    # Connect to iRacing
    print("\n🔌 Connecting to iRacing for capture...")
    if not iracing.startup():
        print(
            "❌ iRacing not found. Make sure iRacing is running with an active session."
        )
        return

    print("✅ Connected! Capturing telemetry...\n")
    session_dir = capture.start_session()

    tick_rate_hz = config.get("iracing", {}).get("tick_rate_hz", 30)
    tick_interval = 1.0 / tick_rate_hz

    try:
        while True:
            if iracing.is_connected:
                telemetry = iracing.get_telemetry()
                session_info = iracing.get_session_info()

                # Read telemetry units for unit conversion (kPa, degF, etc.)
                try:
                    units = iracing.get_telemetry_units()
                except Exception:
                    units = {}

                if capture.should_capture():
                    driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
                    filepath = capture.capture_snapshot(
                        telemetry, session_info, driver_names, units=units
                    )
                    print(
                        f"  📸 Captured snapshot {capture.snapshot_count}: {filepath}"
                    )

                # Still update state for consistency
                state.update(telemetry, session_info, units=units)

            time.sleep(tick_interval)

    except KeyboardInterrupt:
        print(
            f"\n\n📸 Capture complete: {capture.snapshot_count} snapshots saved to {session_dir}"
        )
        iracing.shutdown()


def run_replay_mode(
    config: dict,
    replay_dir: str,
    loop: bool = True,
    question: str | None = None,
    replay_speed: float = 0,
):
    """Replay mode — feed captured data through the pipeline.

    If question is provided, process all snapshots then ask once.
    Otherwise, enter interactive mode where you can type questions at any time.

    replay_speed controls how fast snapshots are fed in interactive mode:
        0 = load all instantly (legacy behaviour), then interactive
        N = feed one snapshot every N seconds, state evolves over time
    """
    context_builder = ContextBuilder(config)
    llm = LLMClient(config)
    executor = ActionExecutor(config=config)  # No iRacing client in replay mode
    state = RaceState(config)

    # Set up TTS for replay mode
    voice_config = config.get("voice", {})
    tts = None
    if voice_config.get("tts", {}).get("enabled", False):
        try:
            tts = TTSClient(config)
            if not tts.is_available:
                tts = None
        except Exception:
            tts = None

    print(f"\n🔄 Loading replay data from {replay_dir}...")
    replay = TelemetryReplay(replay_dir, loop=loop)
    count = replay.load()
    print(f"✅ Loaded {count} snapshots\n")

    # Process snapshots to build state
    # For question mode or instant replay (speed=0), load all upfront.
    # For timed replay (speed>0), load only the first snapshot — the rest
    # are fed one at a time in the interactive loop.
    snapshots_processed = 0
    snapshots_to_preload = count if (question or replay_speed == 0) else 1
    for i in range(snapshots_to_preload):
        snapshot = replay.next_snapshot()
        if snapshot is None:
            break
        telemetry = snapshot.get("telemetry", {})
        session_info = snapshot.get("session_info", {})
        driver_names = snapshot.get("driver_names", {})
        units = snapshot.get("units", {})
        if driver_names:
            driver_names = {int(k): v for k, v in driver_names.items()}
        state.update(telemetry, session_info, driver_names, units=units)
        snapshots_processed += 1

    # Show final state summary
    player = state.player
    flags = ", ".join(state.flags_list) if state.flags_list else "Green"
    print(
        f"📊 State: Lap {player.lap}, P{player.position}, "
        f"Fuel {player.fuel_pct:.0%}, Flags: {flags}"
    )
    if replay_speed > 0 and not question:
        print(
            f"   Processed {snapshots_processed}/{count} snapshots "
            f"(feeding one every {replay_speed}s)"
        )
    else:
        print(f"   Processed {snapshots_processed} snapshots")
    print()

    # If question provided on command line, ask it and exit
    if question:
        messages = context_builder.build_prompt(state.get_snapshot(), question=question)
        response = llm.ask(messages)

        # Log the prompt and response to the LLM query log file
        depth = context_builder.context_depth
        llm_query_logger.info(
            "--- LLM QUERY (replay) ---\n"
            f"Depth: {depth}\n"
            f"Question: {question}\n\n"
            "=== PROMPT SENT ===\n"
            f"{json.dumps(messages, indent=2, ensure_ascii=False)}\n\n"
            "=== RESPONSE ===\n"
            f"{response or '(empty)'}\n"
            "--- END ---"
        )

        clean_text, actions = executor.parse_response(response)
        print("=" * 60)
        print("🏁 RACE ENGINEER")
        print("=" * 60)
        print(clean_text)
        if actions:
            print("\n--- Actions ---")
            results = executor.execute(actions)
            for r in results:
                print(f"  {r}")
        print("=" * 60)
        if tts is not None:
            tts.speak_async(clean_text)
        return

    # Interactive mode — type questions, get answers
    print("=" * 60)
    print("🎙️  Interactive Replay Mode")
    print("=" * 60)
    if replay_speed > 0:
        print(f"Timed replay: feeding one snapshot every {replay_speed}s")
        print("Snapshots will advance automatically. Type at any time.")
    print("Type a question and press Enter to query the race engineer.")
    print("Press Enter with no input for a general strategy query.")
    print("Type 'state' to see current race state.")
    print("Type 'depth minimal/medium/full' to change context depth.")
    print("Type 'next' to advance to the next snapshot immediately.")
    print("Type 'voice' to speak your question via microphone.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    # Set up STT for voice input in replay mode
    stt = None
    if voice_config.get("stt", {}).get("enabled", False):
        try:
            stt = STTClient(config)
            if not stt.is_available:
                stt = None
        except Exception:
            stt = None

    # For timed replay, feed snapshots in a background thread
    replay_stop = threading.Event()
    # Shared state for dynamic prompt — seed from current state
    prompt_state = {
        "lap": state.player.lap,
        "pos": state.player.position,
        "fuel_pct": state.player.fuel_pct,
        "snapshot_num": snapshots_processed,
    }
    last_progress_idx = [0]  # Track last printed progress milestone

    def _replay_feeder():
        """Background thread that feeds snapshots at the configured interval."""
        while not replay_stop.is_set():
            time.sleep(replay_speed)
            if replay_stop.is_set():
                break
            snapshot = replay.next_snapshot()
            if snapshot is None:
                print("\n  🏁 Replay complete — all snapshots consumed.\n")
                replay_stop.set()
                break
            telemetry = snapshot.get("telemetry", {})
            session_info = snapshot.get("session_info", {})
            driver_names = snapshot.get("driver_names", {})
            units = snapshot.get("units", {})
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            state.update(telemetry, session_info, driver_names, units=units)
            p = state.player
            # Update shared prompt state
            prompt_state["lap"] = p.lap
            prompt_state["pos"] = p.position
            prompt_state["fuel_pct"] = p.fuel_pct
            prompt_state["snapshot_num"] = replay._index
            # Print progress at milestones (every ~20% of total snapshots)
            step = max(1, count // 5)
            if (
                replay._index > 1
                and replay._index % step == 0
                and replay._index != last_progress_idx[0]
            ):
                flags = ", ".join(state.flags_list) if state.flags_list else "Green"
                print(
                    f"  📊 Lap {p.lap}, P{p.position}, Fuel {p.fuel_pct:.0%}, {flags} — {replay._index}/{count}"
                )
                last_progress_idx[0] = replay._index

    if replay_speed > 0:
        feeder_thread = threading.Thread(target=_replay_feeder, daemon=True)
        feeder_thread.start()

    while True:
        try:
            # Build dynamic prompt with current state
            lap = prompt_state["lap"]
            pos = prompt_state["pos"]
            fuel_pct = prompt_state["fuel_pct"]
            snap = prompt_state["snapshot_num"]
            prompt_prefix = f"🏎️ [Lap {lap} P{pos} {fuel_pct:.0%}⛽ {snap}/{count}]"
            user_input = input(f"{prompt_prefix} > ").strip()
            # Strip UTF-8 BOM that PowerShell may prepend when piping input
            if user_input.startswith("﻿"):
                user_input = user_input[1:].strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Replay ended.")
            break

        if not user_input:
            # Empty input = general strategy query (like pressing F9 in live mode)
            user_input = "What's my current strategy?"

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        if user_input.lower() == "voice":
            # Voice input mode — record from microphone, transcribe, query LLM
            if stt is None:
                print(
                    "  ❌ Voice input not available. Install voice deps: uv pip install -e '.[voice]'\n"
                )
                continue
            print("  🎤 Recording... Press Enter to stop.")
            stop_event = threading.Event()
            result = [""]

            def _record():
                try:
                    result[0] = stt.listen_push_to_talk(
                        stop_event,
                        max_duration_s=voice_config.get("trigger", {}).get(
                            "max_record_seconds", 15
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Voice recording/transcription failed (replay mode)"
                    )

            rec_thread = threading.Thread(target=_record)
            rec_thread.start()
            input()  # Wait for Enter to stop
            stop_event.set()
            rec_thread.join(timeout=5)

            if result[0].strip():
                print(f'  📝 Heard: "{result[0]}"')
                user_input = result[0]
            else:
                print("  ❌ No speech detected.\n")
                continue

        if user_input.lower() == "next":
            # Advance to the next snapshot immediately
            snapshot = replay.next_snapshot()
            if snapshot is None:
                print("  🏁 No more snapshots to replay.\n")
                continue
            telemetry = snapshot.get("telemetry", {})
            session_info = snapshot.get("session_info", {})
            driver_names = snapshot.get("driver_names", {})
            units = snapshot.get("units", {})
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            state.update(telemetry, session_info, driver_names, units=units)
            p = state.player
            print(
                f"  📊 Lap {p.lap}, P{p.position}, "
                f"Fuel {p.fuel_pct:.0%} — "
                f"snapshot {replay._index}/{count}\n"
            )
            continue

        if not user_input:
            # Empty input = general strategy query (like pressing F9 in live mode)
            user_input = "What's my current strategy?"

        if user_input.lower() == "state":
            # Print current state summary
            p = state.player
            print(
                f"  Track: {state.session.track_name} {state.session.track_config}".strip()
            )
            print(
                f"  Lap: {p.lap}/{p.lap + state.session.laps_remain} | "
                f"Pos: P{p.position} (Class P{p.class_position}) | "
                f"Flags: {flags}"
            )
            print(
                f"  Fuel: {p.fuel_level:.1f}L ({p.fuel_pct:.0%}), "
                f"~{state.fuel_laps_remaining:.1f} laps remaining"
            )
            print(
                f"  Tyres: LF {p.tyres.get('LF', TyreState()).temp_center:.0f}°C / "
                f"RF {p.tyres.get('RF', TyreState()).temp_center:.0f}°C / "
                f"LR {p.tyres.get('LR', TyreState()).temp_center:.0f}°C / "
                f"RR {p.tyres.get('RR', TyreState()).temp_center:.0f}°C"
            )
            if state.nearby_cars:
                print(f"  Nearby: {len(state.nearby_cars)} cars")
                for car in state.nearby_cars[:3]:
                    gap_str = (
                        f"+{car.gap_seconds:.1f}s"
                        if car.gap_seconds >= 0
                        else f"{car.gap_seconds:.1f}s"
                    )
                    print(f"    P{car.position} ({car.driver_name}): {gap_str}")
            print()
            continue

        if user_input.lower().startswith("depth "):
            new_depth = user_input.lower().split()[1]
            if new_depth in ("minimal", "medium", "full"):
                config.setdefault("prompt", {})["context_depth"] = new_depth
                context_builder = ContextBuilder(config)  # Rebuild with new depth
                print(f"  ✅ Context depth set to: {new_depth}\n")
            else:
                print(
                    f"  ❌ Unknown depth: {new_depth}. Use minimal, medium, or full.\n"
                )
            continue

        # It's a question — send to LLM
        messages = context_builder.build_prompt(
            state.get_snapshot(), question=user_input
        )
        print("  ⏳ Asking race engineer...\n")
        response = llm.ask(messages)

        # Log the prompt and response to the LLM query log file
        depth = context_builder.context_depth
        llm_query_logger.info(
            "--- LLM QUERY (replay interactive) ---\n"
            f"Depth: {depth}\n"
            f"Question: {user_input}\n\n"
            "=== PROMPT SENT ===\n"
            f"{json.dumps(messages, indent=2, ensure_ascii=False)}\n\n"
            "=== RESPONSE ===\n"
            f"{response or '(empty)'}\n"
            "--- END ---"
        )

        if response:
            clean_text, actions = executor.parse_response(response)
            print("  " + "=" * 56)
            print("  🏁 RACE ENGINEER")
            print("  " + "=" * 56)
            for line in clean_text.split("\n"):
                print(f"  {line}")
            if actions:
                print("\n  --- Actions ---")
                results = executor.execute(actions)
                for r in results:
                    print(f"  {r}")
            print("  " + "=" * 56 + "\n")
            if tts is not None:
                tts.speak_async(clean_text)
        else:
            print("  ❌ No response from LLM.\n")

    # Clean up replay feeder thread if running
    if replay_speed > 0:
        replay_stop.set()
        feeder_thread.join(timeout=2)


def run_generate_samples(config: dict):
    """Generate sample data for testing without iRacing."""
    output_dir = config.get("capture", {}).get("output_dir", "./tests/sample_data")
    print(f"\n📊 Generating sample data in {output_dir}...")
    session_dir = create_sample_data(output_dir)
    print(f"✅ Sample data created in {session_dir}")
    print("\nTo replay, run:")
    print(f"  python main.py --replay {session_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="iRaceEngineer — Real-time iRacing data + LLM strategy"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture mode — record telemetry to JSON files",
    )
    parser.add_argument(
        "--capture-dir", help="Directory to save captured data (default: from config)"
    )
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=1000,
        help="Capture interval in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--replay", help="Replay mode — feed captured data from this directory"
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0,
        help="Seconds between replay snapshots (0 = load all instantly, e.g. 0.1 = ~3min for 1920 snapshots, default: 0)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Don't loop replay when all snapshots are consumed",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample telemetry data for testing",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Capture real iRacing data to tests/sample_data/ (like --capture but saves to the sample data folder for replay/testing)",
    )
    parser.add_argument(
        "--depth",
        choices=["minimal", "medium", "full"],
        help="Override context depth for this run",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Ask a specific question (works with --replay or live mode)",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice input/output (overrides voice.stt.enabled / voice.tts.enabled in config)",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice input/output for this run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    # Generate session timestamp for log file names
    session_ts = _session_timestamp()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # Set up log files (each gets a session timestamp suffix)
    setup_llm_query_log(config, session_ts)
    setup_console_log(config, session_ts)

    # Override context depth if specified
    if args.depth:
        config.setdefault("prompt", {})["context_depth"] = args.depth

    # Override voice settings from CLI flags
    if args.voice or args.no_voice:
        config.setdefault("voice", {}).setdefault("stt", {})["enabled"] = (
            args.voice and not args.no_voice
        )
        config.setdefault("voice", {}).setdefault("tts", {})["enabled"] = (
            args.voice and not args.no_voice
        )

    # Set API key from config if not in environment
    llm_config = config.get("llm", {})
    api_key_env = llm_config.get("api_key_env", "OLLAMA_API_KEY")
    if not os.environ.get(api_key_env):
        # Try common env var names
        for env_var in ["OLLAMA_API_KEY", "OPENAI_API_KEY", "API_KEY"]:
            if os.environ.get(env_var):
                break

    print("=" * 60)
    print("🏎️  iRaceEngineer")
    print("=" * 60)

    if args.generate_samples:
        run_generate_samples(config)
    elif args.save_samples:
        sample_dir = config.get("capture", {}).get("output_dir", "./tests/sample_data")
        run_capture_mode(config, sample_dir, args.capture_interval)
    elif args.capture:
        capture_dir = args.capture_dir or config.get("capture", {}).get(
            "output_dir", "./tests/sample_data"
        )
        run_capture_mode(config, capture_dir, args.capture_interval)
    elif args.replay:
        run_replay_mode(
            config,
            args.replay,
            loop=not args.no_loop,
            question=args.question,
            replay_speed=args.replay_speed,
        )
    else:
        run_live_mode(config, config.get("iracing", {}).get("tick_rate_hz", 30))


if __name__ == "__main__":
    main()
