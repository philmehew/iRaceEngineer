"""
iRaceEngineer — Real-time iRacing data collection with on-demand LLM strategy.

Collects telemetry from iRacing via shared memory (pyirsdk), maintains
an in-memory race state, and sends a condensed snapshot to an OpenAI-compatible
LLM when a button is pressed.

Entry point only — all library code lives in the `modules/` package. The three
run modes (live / capture / replay) are in modules/live_mode.py,
modules/capture_mode.py, and modules/replay_mode.py.

Usage:
    python main.py                    # Normal mode — connect to iRacing
    python main.py --capture          # Capture mode — record telemetry to JSON
    python main.py --replay <dir>     # Replay mode — feed captured data through pipeline
    python main.py --generate-samples # Generate sample data for testing
"""

import argparse
import atexit
import logging
import os
import sys
import yaml

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from modules.capture_mode import run_capture_mode, run_generate_samples
from modules.live_mode import run_live_mode
from modules.logging_setup import (
    _session_timestamp,
    setup_console_log,
    setup_llm_query_log,
)
from modules.replay_mode import run_replay_mode
from modules.single_instance import SingleInstanceGuard


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("iraceengineer")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config or {}


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
        default=None,
        help="Capture interval in milliseconds (default: capture.interval_ms from config)",
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
        "--force",
        action="store_true",
        help="Start even if another iRaceEngineer instance is running",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    # Refuse to run alongside another instance — two instances would fight over
    # the wheel, audio devices, and LLM queries. Lock is OS-held, so a crashed
    # instance never blocks a restart.
    guard = SingleInstanceGuard()
    if not args.force and not guard.acquire():
        print("=" * 60)
        print("❌  iRaceEngineer is already running (another instance holds the lock).")
        print("    If you're sure it isn't, or you want a second instance anyway,")
        print("    rerun with --force.")
        print("=" * 60)
        sys.exit(1)
    atexit.register(guard.release)

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

    print("=" * 60)
    print("🏎️  iRaceEngineer")
    print("=" * 60)

    if args.generate_samples:
        run_generate_samples(config)
    elif args.save_samples:
        sample_dir = config.get("capture", {}).get("output_dir", "./tests/sample_data")
        run_capture_mode(config, sample_dir)
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
        run_live_mode(
            config,
            config.get("iracing", {}).get("tick_rate_hz", 30),
            question=args.question,
        )


if __name__ == "__main__":
    main()
