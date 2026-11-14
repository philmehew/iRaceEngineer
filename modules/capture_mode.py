"""
Capture mode — record live telemetry snapshots to JSON for replay/testing.

Also hosts run_generate_samples (fake Spa 24h stint data via capture.create_sample_data).
"""

import logging
import time

from modules.capture import TelemetryCapture, create_sample_data
from modules.iracing_client import IRacingClient
from modules.race_state import RaceState

logger = logging.getLogger("iraceengineer")


def run_capture_mode(config: dict, capture_dir: str, interval_ms: int | None = None):
    """Capture mode — record telemetry snapshots to JSON files."""
    if interval_ms is None:
        interval_ms = config.get("capture", {}).get("interval_ms", 1000)
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


def run_generate_samples(config: dict):
    """Generate sample data for testing without iRacing."""
    output_dir = config.get("capture", {}).get("output_dir", "./tests/sample_data")
    print(f"\n📊 Generating sample data in {output_dir}...")
    session_dir = create_sample_data(output_dir)
    print(f"✅ Sample data created in {session_dir}")
    print("\nTo replay, run:")
    print(f"  python main.py --replay {session_dir}")
