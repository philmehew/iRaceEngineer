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
from logging.handlers import RotatingFileHandler

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iraceengineer")

# Dedicated logger for LLM query data — writes prompt/response to a local file
llm_query_logger = logging.getLogger("iraceengineer.llm_query")
llm_query_logger.propagate = False  # Don't double-print to console

# Thread safety for live mode: protects state reads/writes between the main
# poll loop and the keyboard-hook thread, and prevents double-press.
_state_lock = threading.Lock()
_llm_in_progress = threading.Event()


def setup_llm_query_log(config: dict):
    """Set up the LLM query log file handler.

    Creates a RotatingFileHandler that writes the prompt and response
    for every LLM call to a local log file. Defaults to logs/llm_queries.log
    with 5 rotating files of 1 MB each.
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("llm_query_log_dir", "logs")
    log_file = log_config.get("llm_query_log_file", "llm_queries.log")
    max_bytes = log_config.get("llm_query_max_bytes", 1_000_000)
    backup_count = log_config.get("llm_query_backup_count", 5)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s\n%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    llm_query_logger.addHandler(handler)
    llm_query_logger.setLevel(logging.INFO)
    logger.info(f"LLM query log: {log_path}")


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
):
    """Handle a button press — build context, call LLM, process response.

    Thread-safe: acquires the state lock to read a consistent snapshot,
    and guards against double-press (skips if an LLM call is already in progress).
    """
    # Guard against double-press — skip if an LLM call is already running
    if _llm_in_progress.is_set():
        logger.info("LLM call already in progress — skipping")
        return
    _llm_in_progress.set()

    try:
        logger.info("Button pressed — querying LLM...")

        # Get a consistent snapshot under the state lock
        with _state_lock:
            snapshot = state.get_snapshot()

        messages = context_builder.build_prompt(snapshot, question=question)

        # Log the context being sent
        depth = context_builder.context_depth
        logger.info(
            f"Sending context (depth={depth}, ~{len(str(messages[-1]['content']))} bytes) to LLM..."
        )

        # Call LLM
        response_text = llm.ask(messages)

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
            return

        # Parse actions from response
        clean_text, actions = executor.parse_response(response_text)

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
        print("=" * 60 + "\n")

    finally:
        _llm_in_progress.clear()


def run_live_mode(config: dict, tick_rate_hz: int = 30):
    """Main loop — connect to iRacing and poll telemetry."""
    iracing = IRacingClient()
    state = RaceState(config)
    context_builder = ContextBuilder(config)
    llm = LLMClient(config)
    executor = ActionExecutor(iracing, config)

    # Set up keyboard trigger
    trigger_key = config.get("trigger", {}).get("key", "f9")
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
    state.set_team_indices(team_indices)
    driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
    state.set_driver_names(driver_names)
    print(
        f"👥 Team detected: {len(team_indices)} cars — {', '.join(driver_names.get(i, f'Car #{i}') for i in team_indices)}\n"
    )

    # Register keyboard hook
    if keyboard_available:
        import keyboard as kb

        kb.add_hotkey(
            trigger_key,
            lambda: handle_button_press(state, context_builder, llm, executor),
        )
        print(f"🎧 Listening for {trigger_key.upper()} key press...")
        print(f"   Press {trigger_key.upper()} to ask the race engineer a question.")
        print("   Press Ctrl+C to exit.\n")

    # Main poll loop
    tick_interval = 1.0 / tick_rate_hz
    try:
        while True:
            if iracing.is_connected:
                # Read telemetry
                telemetry = iracing.get_telemetry()
                session_info = iracing.get_session_info()

                # Update race state (under lock to prevent stale reads from F9 handler)
                driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
                with _state_lock:
                    state.update(telemetry, session_info, driver_names)

            else:
                # iRacing disconnected — try to reconnect
                logger.warning("iRacing disconnected, attempting reconnect...")
                if not iracing.startup():
                    time.sleep(2)
                    continue
                print("✅ Reconnected to iRacing")

            time.sleep(tick_interval)

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
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

                if capture.should_capture():
                    driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
                    filepath = capture.capture_snapshot(
                        telemetry, session_info, driver_names
                    )
                    print(
                        f"  📸 Captured snapshot {capture.snapshot_count}: {filepath}"
                    )

                # Still update state for consistency
                state.update(telemetry, session_info)

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
    replay_speed: int = 0,
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
        if driver_names:
            driver_names = {int(k): v for k, v in driver_names.items()}
        state.update(telemetry, session_info, driver_names)
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
    print("Type 'quit' or press Ctrl+C to exit.\n")

    # For timed replay, feed snapshots in a background thread
    replay_stop = threading.Event()

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
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            state.update(telemetry, session_info, driver_names)
            p = state.player
            print(
                f"  📊 Lap {p.lap}, P{p.position}, "
                f"Fuel {p.fuel_pct:.0%} — "
                f"snapshot {replay._index}/{count}"
            )

    if replay_speed > 0:
        feeder_thread = threading.Thread(target=_replay_feeder, daemon=True)
        feeder_thread.start()

    while True:
        try:
            user_input = input("🏎️ > ").strip()
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

        if user_input.lower() == "next":
            # Advance to the next snapshot immediately
            snapshot = replay.next_snapshot()
            if snapshot is None:
                print("  🏁 No more snapshots to replay.\n")
                continue
            telemetry = snapshot.get("telemetry", {})
            session_info = snapshot.get("session_info", {})
            driver_names = snapshot.get("driver_names", {})
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            state.update(telemetry, session_info, driver_names)
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

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

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
        type=int,
        default=0,
        help="Seconds between replay snapshots (0 = load all instantly, default: 0)",
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
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # Set up LLM query log file
    setup_llm_query_log(config)

    # Override context depth if specified
    if args.depth:
        config.setdefault("prompt", {})["context_depth"] = args.depth

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
