"""
Capture and replay telemetry for testing without iRacing running.

Capture mode (--capture): records telemetry snapshots from a live iRacing
session to timestamped JSON files.

Replay mode (--replay): reads captured JSON files and feeds them through
the pipeline instead of live iRacing data.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TelemetryCapture:
    """Records telemetry snapshots to JSON for later replay."""

    def __init__(self, output_dir: str, interval_ms: int = 1000):
        """
        Args:
            output_dir: Directory to save captured snapshots.
            interval_ms: Minimum interval between snapshots in milliseconds.
        """
        self.output_dir = Path(output_dir)
        self.interval_ms = interval_ms
        self.interval_s = interval_ms / 1000.0
        self._last_capture_time = 0.0
        self._snapshot_count = 0

    def start_session(self) -> str:
        """Create a new session directory for capturing.

        Returns:
            Path to the session directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_count = 0
        logger.info(f"Capture session started: {self.session_dir}")
        return str(self.session_dir)

    def should_capture(self) -> bool:
        """Check if enough time has elapsed since the last capture."""
        now = time.time()
        if now - self._last_capture_time >= self.interval_s:
            self._last_capture_time = now
            return True
        return False

    def capture_snapshot(
        self,
        telemetry: dict,
        session_info: dict,
        driver_names: dict | None = None,
        units: dict | None = None,
    ) -> str:
        """Save a telemetry snapshot to a JSON file.

        Args:
            telemetry: Raw telemetry dict from IRacingClient.get_telemetry()
            session_info: Parsed session info dict
            driver_names: Optional driver name lookup {car_idx: name}
            units: Optional telemetry unit mapping from IRacingClient.get_telemetry_units()

        Returns:
            Path to the saved file.
        """
        self._snapshot_count += 1
        filename = f"snapshot_{self._snapshot_count:04d}.json"

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "snapshot_num": self._snapshot_count,
            "session_info": session_info,
            "telemetry": telemetry,
            "driver_names": driver_names or {},
            "units": units or {},
        }

        filepath = self.session_dir / filename
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.debug(f"Captured snapshot {self._snapshot_count} to {filepath}")
        return str(filepath)

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count


class TelemetryReplay:
    """Replays captured telemetry for testing without iRacing."""

    def __init__(self, capture_dir: str, loop: bool = True):
        """
        Args:
            capture_dir: Directory containing captured snapshot JSON files.
            loop: If True, loop back to the start when all snapshots are replayed.
        """
        self.capture_dir = Path(capture_dir)
        self.loop = loop
        self._snapshots: list[dict] = []
        self._index = 0
        self._loaded = False

    def load(self) -> int:
        """Load all snapshot files from the capture directory.

        Returns:
            Number of snapshots loaded.
        """
        if not self.capture_dir.exists():
            raise FileNotFoundError(f"Capture directory not found: {self.capture_dir}")

        # Find and sort all snapshot files
        files = sorted(self.capture_dir.glob("snapshot_*.json"))
        if not files:
            raise ValueError(f"No snapshot files found in {self.capture_dir}")

        self._snapshots = []
        for filepath in files:
            try:
                with open(filepath) as f:
                    snapshot = json.load(f)
                self._snapshots.append(snapshot)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping invalid snapshot file {filepath}: {e}")

        self._loaded = True
        self._index = 0
        logger.info(f"Loaded {len(self._snapshots)} snapshots from {self.capture_dir}")
        return len(self._snapshots)

    def next_snapshot(self) -> dict | None:
        """Get the next snapshot in sequence.

        Returns:
            Snapshot dict, or None if all snapshots have been replayed
            (and loop is False).
        """
        if not self._loaded:
            self.load()

        if self._index >= len(self._snapshots):
            if self.loop:
                self._index = 0
                logger.info("Replay loop: restarting from beginning")
            else:
                return None

        snapshot = self._snapshots[self._index]
        self._index += 1
        return snapshot

    def has_more(self) -> bool:
        """Check if there are more snapshots to replay."""
        if not self._loaded:
            return True
        return self._index < len(self._snapshots) or self.loop

    @property
    def progress(self) -> tuple[int, int]:
        """Return (current_index, total_snapshots)."""
        return (self._index, len(self._snapshots))

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_sample_data(output_dir: str) -> str:
    """Generate sample telemetry data for testing without iRacing.

    Creates a set of realistic snapshots that simulate a race scenario.

    Args:
        output_dir: Directory to save the sample data.

    Returns:
        Path to the session directory.
    """
    capture = TelemetryCapture(output_dir)
    session_dir = capture.start_session()

    # Simulate a 10-lap race stint with realistic progression
    # Fuel burn: ~3.8L/lap, starting from 110L full tank
    # Tyre degradation: gradual increase in wear and temps
    # Lap times: baseline 92.5s + 0.15s/lap degradation
    for lap in range(1, 11):
        # Simulate progressive tyre deg and fuel burn
        base_temp = round(85 + lap * 1.2, 1)
        fuel_start = round(110 - (lap - 1) * 3.8, 1)
        fuel_end = round(fuel_start - 3.8, 1)
        lap_time = round(92.5 + lap * 0.15, 3)
        fuel_pct = round(fuel_end / 110, 4)

        # Opponents vary slightly — Wayne pushes, Andre is consistent
        wayne_last = round(92.1 + lap * 0.08, 3)
        andre_last = round(92.4 + lap * 0.12, 3)
        dan_last = round(92.8 + lap * 0.10, 3)
        dave_last = round(92.6 + lap * 0.11, 3)  # noqa: F841 — used in LastLapTime array below
        liam_last = round(93.0 + lap * 0.09, 3)
        david_last = round(93.2 + lap * 0.13, 3)

        # Gap to car ahead (P5 Andre) — fluctuates, closes over stint
        gap_ahead = round(2.1 - lap * 0.05 + (0.1 if lap % 3 == 0 else 0), 1)
        gap_behind = round(-1.8 - lap * 0.03, 1)

        snapshot = {
            "timestamp": f"2024-06-08T14:30:{lap:02d}.000",
            "snapshot_num": lap,
            "session_info": {
                "WeekendInfo": {
                    "TrackName": "Circuit de Spa-Francorchamps",
                    "TrackConfigName": "Grand Prix",
                },
                "SessionInfo": {
                    "Sessions": [
                        {
                            "SessionNum": 0,
                            "SessionType": "Race",
                            "SessionName": "Race",
                        }
                    ]
                },
                "DriverInfo": {
                    "Drivers": [
                        {
                            "CarIdx": 0,
                            "UserName": "Patrik Farsang",
                            "CarNumber": "7",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 1,
                            "UserName": "Wayne Smith8",
                            "CarNumber": "4",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 2,
                            "UserName": "Andre Groove",
                            "CarNumber": "12",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 3,
                            "UserName": "Dan Golden",
                            "CarNumber": "23",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 4,
                            "UserName": "Dave Cartnerdobbs",
                            "CarNumber": "55",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 5,
                            "UserName": "Liam Biggs",
                            "CarNumber": "88",
                            "TeamName": "iRaceEngineer",
                        },
                        {
                            "CarIdx": 6,
                            "UserName": "David Barlow",
                            "CarNumber": "31",
                            "TeamName": "iRaceEngineer",
                        },
                    ]
                },
                "SplitTimeInfo": {
                    "Sectors": [
                        {"SectorNum": 1, "SectorStartPct": 0.0},
                        {"SectorNum": 2, "SectorStartPct": 0.3333},
                        {"SectorNum": 3, "SectorStartPct": 0.6667},
                    ]
                },
            },
            "telemetry": {
                # Session
                "SessionFlags": 4,  # Green
                "SessionLapsRemain": 60 - lap,
                "SessionTimeRemain": 3600 - lap * 90,
                "SessionNum": 0,
                "SessionState": 4,  # Racing
                "RaceLaps": lap,
                # Player car
                "PlayerCarIdx": 0,
                "PlayerCarPosition": 4,
                "PlayerCarClassPosition": 2,
                "Lap": lap,
                "LapCompleted": lap - 1,
                "LapDistPct": 0.5,
                "Speed": round(85.0 + (10 - lap) * 0.3, 1),
                "RPM": 8200,
                "Gear": 5,
                "Throttle": 0.95,
                "Brake": 0.0,
                # Fuel
                "FuelLevel": fuel_end,
                "FuelLevelPct": fuel_pct,
                "FuelUsePerHour": 28.5,
                # Laps
                "LapCurrentLapTime": round(lap_time * 0.5, 3),
                "LapBestLapTime": 92.5,
                "LapLastLapTime": lap_time,
                "LapDeltaToBestLap": round(lap_time - 92.5, 3),
                # Tyres — temps rise and degrade over the stint
                "LFtempCL": round(base_temp - 2, 1),
                "LFtempCM": round(base_temp, 1),
                "LFtempCR": round(base_temp + 3, 1),
                "RFtempCL": round(base_temp - 1, 1),
                "RFtempCM": round(base_temp + 2, 1),
                "RFtempCR": round(base_temp + 5, 1),
                "LRtempCL": round(base_temp + 5, 1),
                "LRtempCM": round(base_temp + 8, 1),
                "LRtempCR": round(base_temp + 12, 1),
                "RRtempCL": round(base_temp + 6, 1),
                "RRtempCM": round(base_temp + 10, 1),
                "RRtempCR": round(base_temp + 14, 1),
                "LFcoldPressure": 26.5,
                "RFcoldPressure": 26.3,
                "LRcoldPressure": 24.8,
                "RRcoldPressure": 24.5,
                # Tyre wear — rear degrades faster, rounded to avoid FP noise
                "LFwearL": round(lap * 0.008, 4),
                "LFwearM": round(lap * 0.01, 4),
                "LFwearR": round(lap * 0.012, 4),
                "RFwearL": round(lap * 0.007, 4),
                "RFwearM": round(lap * 0.009, 4),
                "RFwearR": round(lap * 0.011, 4),
                "LRwearL": round(lap * 0.010, 4),
                "LRwearM": round(lap * 0.014, 4),
                "LRwearR": round(lap * 0.016, 4),
                "RRwearL": round(lap * 0.011, 4),
                "RRwearM": round(lap * 0.015, 4),
                "RRwearR": round(lap * 0.018, 4),
                # Weather — slight track temp rise over the stint
                "TrackTemp": round(27.0 + lap * 0.2, 1),
                "TrackWetness": 0.0,
                "AirTemp": 22.0,
                "AirPressure": 1013.0,
                "Precipitation": 0.0,
                "WindDir": 180.0,
                "WindVel": 3.5,
                # Pit
                "OnPitRoad": False,
                "PitstopActive": False,
                "PitsOpen": True,
                "FastRepairAvailable": True,
                "TireSetsAvailable": 3,
                "TireSetsUsed": 1,
                "PlayerTireCompound": 0,
                # Push to pass — one used by lap 8
                "P2P_Status": 0,
                "P2P_Count": 3 if lap < 8 else 2,
                # Standings (arrays) — 7 team drivers
                "CarIdxPosition": [1, 2, 3, 4, 5, 6, 7],
                "CarIdxClassPosition": [1, 1, 2, 2, 3, 3, 4],
                "CarIdxLap": [lap, lap, lap, lap, lap, lap, lap],
                "CarIdxLapDistPct": [0.52, 0.50, 0.49, 0.46, 0.44, 0.41, 0.38],
                "CarIdxTrackSurface": [2, 2, 2, 2, 2, 2, 2],  # 2 = on track
                "CarIdxOnPitRoad": [False, False, False, False, False, False, False],
                "CarIdxBestLapTime": [91.8, 92.0, 92.1, 92.3, 92.5, 92.7, 93.0],
                "CarIdxLastLapTime": [
                    91.9,
                    wayne_last,
                    andre_last,
                    lap_time,
                    dan_last,
                    liam_last,
                    david_last,
                ],
                "CarIdxP2P_Status": [0, 0, 1, 0, 0, 1, 0],
                "CarIdxP2P_Count": [2, 4, 1, 3 if lap < 8 else 2, 2, 3, 1],
                "CarIdxTireCompound": [0, 0, 1, 0, 0, 1, 0],
                # Incidents
                "CarIdxSessionFlags": [0, 0, 0, 0],
                "PlayerCarMyIncidentCount": 0,
                "PlayerCarTeamIncidentCount": 2,
                "CarDistAhead": gap_ahead,
                "CarDistBehind": gap_behind,
            },
            "driver_names": {
                "0": "Patrik Farsang",
                "1": "Wayne Smith8",
                "2": "Andre Groove",
                "3": "Dan Golden",
                "4": "Dave Cartnerdobbs",
                "5": "Liam Biggs",
                "6": "David Barlow",
            },
        }

        capture.capture_snapshot(
            telemetry=snapshot["telemetry"],  # type: ignore[arg-type]
            session_info=snapshot["session_info"],  # type: ignore[arg-type]
            driver_names=snapshot["driver_names"],  # type: ignore[arg-type]
        )

    logger.info(
        f"Created sample data with {capture.snapshot_count} snapshots in {session_dir}"
    )
    return session_dir
