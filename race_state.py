"""
In-memory race state model — maintains a queryable model of the current race,
updated each tick, with per-lap history for trend analysis.

Architecture:
- DriverState: Universal per-car class used for ALL cars (player, teammates, nearby)
  Player-only fields (fuel, tyre temps/wear, brakes, throttle) are left at defaults for other cars.
- RaceState: Holds one DriverState for the player, plus dicts for teammates and nearby cars.
  Shared session info and standings are stored once.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TyreState:
    """Tyre data for one corner."""

    temp_left: float = 0.0  # Inner/cold side temp (°C)
    temp_center: float = 0.0  # Center temp (°C)
    temp_right: float = 0.0  # Outer/hot side temp (°C)
    cold_pressure: float = 0.0
    wear_left: float = 0.0  # 0-1 (fraction, not percentage)
    wear_center: float = 0.0
    wear_right: float = 0.0


@dataclass
class LapRecord:
    """Per-lap historical data."""

    lap_number: int = 0
    lap_time: float = -1.0
    fuel_at_start: float = 0.0
    fuel_at_end: float = 0.0
    fuel_used: float = 0.0
    tyre_temps: dict[str, TyreState] = field(default_factory=dict)
    was_personal_best: bool = False
    was_fastest: bool = False


@dataclass
class DriverState:
    """Per-driver state — used for the player, teammates, AND nearby cars.

    Fields that are player-only (fuel detail, tyre temps/wear, brake inputs)
    are left at default values (0.0, False, etc.) for non-player cars.
    This keeps the class universal while making it clear what data is available.
    """

    # Identity
    car_idx: int = 0
    driver_name: str = ""

    # Position & lap
    position: int = 0
    class_position: int = 0
    lap: int = 0
    lap_completed: int = 0
    lap_dist_pct: float = 0.0
    speed: float = 0.0  # m/s (player only — CarIdxEstTime for others)
    rpm: float = 0.0
    gear: int = 0

    # Player-only: inputs
    throttle: float = 0.0
    brake: float = 0.0

    # Player-only: engine health
    oil_temp: float = 0.0  # °C
    oil_press: float = 0.0  # bar/PSI
    oil_level: float = 0.0
    water_temp: float = 0.0  # °C
    water_level: float = 0.0
    fuel_press: float = 0.0
    engine_warnings: int = 0  # Bitmask
    manifold_press: float = 0.0
    voltage: float = 0.0

    # Player-only: fuel
    fuel_level: float = 0.0  # Litres
    fuel_pct: float = 0.0  # 0-1
    fuel_use_per_hour: float = 0.0  # L/hr
    fuel_laps_remaining: float = 0.0  # Derived

    # Tyres (per corner: LF, RF, LR, RR)
    # Player-only: temps, pressures, and wear
    # For other cars: only compound is available
    tyres: dict[str, TyreState] = field(default_factory=dict)
    tire_compound: int = 0
    # Player-only: per-corner odometer (km driven on that tyre)
    tyre_odometers: dict[str, float] = field(default_factory=dict)

    # Player-only: brakes
    brake_pressures: dict[str, float] = field(default_factory=dict)
    brake_abs_active: bool = False
    brake_bias: float = 0.0  # dcBrakeBias

    # Player-only: G-forces
    lat_accel: float = 0.0  # Lateral g-force
    long_accel: float = 0.0  # Longitudinal g-force
    vert_accel: float = 0.0  # Vertical g-force

    # Player-only: damage and penalties
    weight_penalty: float = 0.0  # kg added for damage
    fast_repairs_used: int = 0
    pit_repair_time_left: float = 0.0  # Repair time remaining in pits
    pit_opt_repair_time_left: float = 0.0  # Optimal repair time remaining

    # Player-only: car status
    is_on_track: bool = False
    is_in_garage: bool = False

    # Player-only: proximity
    car_left_right: int = 0  # 0=none, 1=car left, 2=car right, 3=both
    car_dist_ahead: float = 0.0  # metres
    car_dist_behind: float = 0.0  # metres
    tow_time: float = 0.0  # seconds of tow available

    # Player-only: shift lights
    shift_indicator_pct: float = 0.0
    shift_rpm: float = 0.0

    # Lap times
    current_lap_time: float = -1.0
    best_lap_time: float = -1.0
    last_lap_time: float = -1.0
    delta_to_best: float = 0.0
    lap_time_trend: float = 0.0  # Derived: seconds/lap degradation

    # Per-lap history (player and teammates only)
    lap_history: list[LapRecord] = field(default_factory=list)

    # Damage/Incidents (player only)
    incidents: int = 0
    team_incidents: int = 0

    # Pit status
    on_pit_road: bool = False
    pitstop_active: bool = False
    pits_open: bool = False
    in_pit_stall: bool = False
    fast_repair_available: bool = False

    # Push-to-pass
    p2p_active: bool = False
    p2p_remaining: int = 0

    # Tyre sets (player only)
    tire_sets_available: int = 0
    tire_sets_used: int = 0

    # Gap to player (derived, for nearby cars)
    gap_seconds: float = 0.0  # Positive = ahead of player, negative = behind

    # Track surface
    track_surface: int = 0  # 0=off, 1=approaching, 2=on track


@dataclass
class SessionState:
    """Current session-level state (shared across all drivers)."""

    track_name: str = ""
    track_config: str = ""
    session_type: str = ""
    session_name: str = ""
    session_num: int = 0
    laps_total: int = 0
    laps_remain: int = 0
    time_remain: float = 0.0  # Seconds
    flags: int = 0  # SessionFlags bitmask
    session_state: int = 0
    race_laps: int = 0

    # Weather (shared across all cars)
    track_temp: float = 0.0  # °C
    track_wetness: float = 0.0  # 0-1
    weather_declared_wet: bool = False  # iRacing declared wet conditions
    air_temp: float = 0.0  # °C
    air_pressure: float = 0.0
    precipitation: float = 0.0  # 0-1
    wind_dir: float = 0.0
    wind_vel: float = 0.0
    skies: int = 0

    # Session config (from WeekendInfo / DriverInfo)
    fuel_max_litres: float = 0.0  # Tank capacity
    is_fixed_setup: bool = False
    incident_limit: str = ""  # e.g. "unlimited", "17x"
    fast_repairs_limit: str = ""  # e.g. "unlimited"
    track_num_turns: int = 0
    track_length_km: float = 0.0
    pit_speed_limit_kph: float = 0.0
    num_starters: int = 0
    car_class_name: str = ""
    # Shift RPMs
    idle_rpm: float = 0.0
    redline_rpm: float = 0.0
    shift_rpm: float = 0.0
    shift_first_rpm: float = 0.0
    shift_last_rpm: float = 0.0
    shift_blink_rpm: float = 0.0


# Session flag constants (from iRacing SDK)
FLAG_GREEN = 0x00000001
FLAG_YELLOW = 0x00000002
FLAG_RED = 0x00000004
FLAG_CHECKERED = 0x00000008
FLAG_WHITE = 0x00000010
FLAG_BLACK = 0x00000020
FLAG_DISQUALIFIED = 0x00000040
FLAG_BLUE = 0x00008000
FLAG_RESTART = 0x00080000


class RaceState:
    """Maintains the current race state, updated each tick.

    Uses DriverState for ALL cars — player, teammates, and nearby.
    Player-only fields (fuel, tyre temps, brakes) are populated for the player
    and left at defaults for other cars.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.session = SessionState()
        self.player = DriverState(car_idx=0, driver_name="Player")
        self.team_drivers: dict[int, DriverState] = {}  # car_idx -> DriverState
        self.nearby_cars: list[DriverState] = []  # Cars near the player
        self.standings: list[dict] = []  # All cars, basic info

        # Internal tracking for lap detection
        self._last_lap_completed: int = -1
        self._fuel_at_lap_start: float = 0.0

        # Timing
        self._last_update_time: float = 0.0
        self._tick_count: int = 0

        # Driver name lookup (set by main from iracing_client)
        self._driver_names: dict[int, str] = {}

        # Team car indices (populated when session info is available)
        self._team_car_indices: set[int] = set()

    def set_driver_names(self, names: dict[int, str]):
        """Set driver name lookup from IRacingClient."""
        self._driver_names = names

    def set_team_indices(self, indices: set[int]):
        """Set which car indices are team drivers (for full tracking)."""
        self._team_car_indices = indices

    def update(
        self,
        telemetry: dict,
        session_info: dict,
        driver_names: dict[int, str] | None = None,
    ):
        """Update race state from telemetry data and session info.

        Called each tick from the main poll loop.
        """
        if driver_names:
            self._driver_names = driver_names

        self._tick_count += 1
        self._last_update_time = time.time()

        # Update session state
        self._update_session(telemetry, session_info)

        # Update player state
        self._update_player(telemetry)

        # Update nearby cars / standings
        self._update_standings(telemetry)

        # Check for lap completion (to record lap history)
        self._check_lap_completion(telemetry)

    def _update_session(self, telemetry: dict, session_info: dict):
        """Update session-level state from telemetry and parsed session info."""
        self.session.flags = int(telemetry.get("SessionFlags", 0))
        self.session.laps_remain = int(telemetry.get("SessionLapsRemain", 0))
        self.session.time_remain = float(telemetry.get("SessionTimeRemain", 0.0))
        self.session.session_num = int(telemetry.get("SessionNum", 0))
        self.session.session_state = int(telemetry.get("SessionState", 0))
        self.session.race_laps = int(telemetry.get("RaceLaps", 0))

        # Weather
        self.session.track_temp = float(telemetry.get("TrackTemp", 0.0))
        self.session.track_wetness = float(telemetry.get("TrackWetness", 0.0))
        self.session.weather_declared_wet = bool(
            telemetry.get("WeatherDeclaredWet", False)
        )
        self.session.air_temp = float(telemetry.get("AirTemp", 0.0))
        self.session.air_pressure = float(telemetry.get("AirPressure", 0.0))
        self.session.precipitation = float(telemetry.get("Precipitation", 0.0))
        self.session.wind_dir = float(telemetry.get("WindDir", 0.0))
        self.session.wind_vel = float(telemetry.get("WindVel", 0.0))
        self.session.skies = int(telemetry.get("Skies", 0))

        # From session info (less frequent updates)
        if session_info:
            weekend = session_info.get("WeekendInfo", {})
            self.session.track_name = weekend.get("TrackName", self.session.track_name)
            self.session.track_config = weekend.get(
                "TrackConfigName", self.session.track_config
            )
            # Parse track length — may be "5.7961 km" or numeric
            track_length_raw = weekend.get("TrackLength", 0)
            if isinstance(track_length_raw, str):
                self.session.track_length_km = float(
                    track_length_raw.replace(" km", "").replace(" km", "")
                )
            else:
                self.session.track_length_km = float(track_length_raw or 0)
            self.session.track_num_turns = int(weekend.get("TrackNumTurns", 0))
            # Parse pit speed limit — may be "60.00 kph" or numeric
            pit_speed_raw = weekend.get("TrackPitSpeedLimit", 0)
            if isinstance(pit_speed_raw, str):
                self.session.pit_speed_limit_kph = float(
                    pit_speed_raw.replace(" kph", "")
                )
            else:
                self.session.pit_speed_limit_kph = float(pit_speed_raw or 0)

            # WeekendOptions
            options = weekend.get("WeekendOptions") or {}
            self.session.is_fixed_setup = bool(int(options.get("IsFixedSetup", 0)))
            self.session.incident_limit = str(options.get("IncidentLimit", ""))
            self.session.fast_repairs_limit = str(options.get("FastRepairsLimit", ""))
            self.session.num_starters = int(
                options.get("NumStarters", weekend.get("MaxDrivers", 0))
            )

            sessions = session_info.get("SessionInfo", {}).get("Sessions", [])
            current_num = self.session.session_num
            for s in sessions:
                if s.get("SessionNum") == current_num:
                    self.session.session_type = s.get(
                        "SessionType", self.session.session_type
                    )
                    self.session.session_name = s.get(
                        "SessionName", self.session.session_name
                    )
                    break

            # Driver info — car spec and shift RPMs
            driver_info = session_info.get("DriverInfo", {})
            self.session.fuel_max_litres = float(
                driver_info.get("DriverCarFuelMaxLtr", 0.0)
            )
            self.session.idle_rpm = float(driver_info.get("DriverCarIdleRPM", 0.0))
            self.session.redline_rpm = float(driver_info.get("DriverCarRedLine", 0.0))
            self.session.shift_rpm = float(driver_info.get("DriverCarSLShiftRPM", 0.0))
            self.session.shift_first_rpm = float(
                driver_info.get("DriverCarSLFirstRPM", 0.0)
            )
            self.session.shift_last_rpm = float(
                driver_info.get("DriverCarSLLastRPM", 0.0)
            )
            self.session.shift_blink_rpm = float(
                driver_info.get("DriverCarSLBlinkRPM", 0.0)
            )

    def _update_player(self, telemetry: dict):
        """Update player car state from telemetry."""
        p = self.player
        p.car_idx = int(telemetry.get("PlayerCarIdx", 0))
        p.driver_name = self._driver_names.get(p.car_idx, "Player")

        # Basic car state
        p.position = int(telemetry.get("PlayerCarPosition", 0))
        p.class_position = int(telemetry.get("PlayerCarClassPosition", 0))
        p.lap = int(telemetry.get("Lap", 0))
        p.lap_completed = int(telemetry.get("LapCompleted", 0))
        p.lap_dist_pct = float(telemetry.get("LapDistPct", 0.0))
        p.speed = float(telemetry.get("Speed", 0.0))
        p.rpm = float(telemetry.get("RPM", 0.0))
        p.gear = int(telemetry.get("Gear", 0))
        p.throttle = float(telemetry.get("Throttle", 0.0))
        p.brake = float(telemetry.get("Brake", 0.0))

        # Engine health (player only)
        p.oil_temp = float(telemetry.get("OilTemp", 0.0))
        p.oil_press = float(telemetry.get("OilPress", 0.0))
        p.oil_level = float(telemetry.get("OilLevel", 0.0))
        p.water_temp = float(telemetry.get("WaterTemp", 0.0))
        p.water_level = float(telemetry.get("WaterLevel", 0.0))
        p.fuel_press = float(telemetry.get("FuelPress", 0.0))
        p.engine_warnings = int(telemetry.get("EngineWarnings", 0))
        p.manifold_press = float(telemetry.get("ManifoldPress", 0.0))
        p.voltage = float(telemetry.get("Voltage", 0.0))

        # Car status
        p.is_on_track = bool(telemetry.get("IsOnTrack", False))
        p.is_in_garage = bool(telemetry.get("IsInGarage", False))

        # G-forces (player only)
        p.lat_accel = float(telemetry.get("LatAccel", 0.0))
        p.long_accel = float(telemetry.get("LongAccel", 0.0))
        p.vert_accel = float(telemetry.get("VertAccel", 0.0))

        # Damage and penalties (player only)
        p.weight_penalty = float(telemetry.get("PlayerCarWeightPenalty", 0.0))
        p.fast_repairs_used = int(telemetry.get("PlayerFastRepairsUsed", 0))
        p.pit_repair_time_left = float(telemetry.get("PitRepairLeft", 0.0))
        p.pit_opt_repair_time_left = float(telemetry.get("PitOptRepairLeft", 0.0))

        # Proximity (player only)
        p.car_left_right = int(telemetry.get("CarLeftRight", 0))
        p.car_dist_ahead = float(telemetry.get("CarDistAhead", 0.0))
        p.car_dist_behind = float(telemetry.get("CarDistBehind", 0.0))
        p.tow_time = float(telemetry.get("PlayerCarTowTime", 0.0))

        # Shift lights (player only)
        p.shift_indicator_pct = float(telemetry.get("ShiftIndicatorPct", 0.0))
        p.shift_rpm = float(telemetry.get("PlayerCarSLShiftRPM", 0.0))

        # Brake bias (player only)
        p.brake_bias = float(telemetry.get("dcBrakeBias", 0.0))

        # Fuel (player only)
        p.fuel_level = float(telemetry.get("FuelLevel", 0.0))
        p.fuel_pct = float(telemetry.get("FuelLevelPct", 0.0))
        p.fuel_use_per_hour = float(telemetry.get("FuelUsePerHour", 0.0))

        # Tyres (per corner — player only for temps/wear)
        for corner in ["LF", "RF", "LR", "RR"]:
            prefix = corner
            p.tyres[corner] = TyreState(
                temp_left=float(telemetry.get(f"{prefix}tempCL", 0.0)),
                temp_center=float(telemetry.get(f"{prefix}tempCM", 0.0)),
                temp_right=float(telemetry.get(f"{prefix}tempCR", 0.0)),
                cold_pressure=float(telemetry.get(f"{prefix}coldPressure", 0.0)),
                wear_left=float(telemetry.get(f"{prefix}wearL", 0.0)),
                wear_center=float(telemetry.get(f"{prefix}wearM", 0.0)),
                wear_right=float(telemetry.get(f"{prefix}wearR", 0.0)),
            )

        # Tyre odometers (per corner — km driven on that tyre)
        p.tyre_odometers = {
            "LF": float(telemetry.get("LFodometer", 0.0)),
            "RF": float(telemetry.get("RFodometer", 0.0)),
            "LR": float(telemetry.get("LRodometer", 0.0)),
            "RR": float(telemetry.get("RRodometer", 0.0)),
        }

        # Brakes (player only)
        p.brake_pressures = {
            "LF": float(telemetry.get("LFbrakeLinePress", 0.0)),
            "RF": float(telemetry.get("RFbrakeLinePress", 0.0)),
            "LR": float(telemetry.get("LRbrakeLinePress", 0.0)),
            "RR": float(telemetry.get("RRbrakeLinePress", 0.0)),
        }
        p.brake_abs_active = bool(telemetry.get("BrakeABSactive", False))

        # Laps
        p.current_lap_time = float(telemetry.get("LapCurrentLapTime", -1.0))
        p.best_lap_time = float(telemetry.get("LapBestLapTime", -1.0))
        p.last_lap_time = float(telemetry.get("LapLastLapTime", -1.0))
        p.delta_to_best = float(telemetry.get("LapDeltaToBestLap", 0.0))

        # Damage/Incidents (player only)
        p.incidents = int(telemetry.get("PlayerCarMyIncidentCount", 0))
        p.team_incidents = int(telemetry.get("PlayerCarTeamIncidentCount", 0))

        # Pit
        p.on_pit_road = bool(telemetry.get("OnPitRoad", False))
        p.pitstop_active = bool(telemetry.get("PitstopActive", False))
        p.pits_open = bool(telemetry.get("PitsOpen", False))
        p.in_pit_stall = bool(telemetry.get("PlayerCarInPitStall", False))
        p.fast_repair_available = bool(telemetry.get("FastRepairAvailable", False))

        # Push-to-pass
        p.p2p_active = bool(telemetry.get("P2P_Status", False))
        p.p2p_remaining = int(telemetry.get("P2P_Count", 0))

        # Tyre sets (player only)
        p.tire_sets_available = int(telemetry.get("TireSetsAvailable", 0))
        p.tire_sets_used = int(telemetry.get("TireSetsUsed", 0))
        p.tire_compound = int(telemetry.get("PlayerTireCompound", 0))

    def _update_standings(self, telemetry: dict):
        """Update standings, nearby cars, and team drivers from telemetry arrays."""
        player_idx = int(telemetry.get("PlayerCarIdx", 0))
        player_lap_dist_pct = float(telemetry.get("LapDistPct", 0.0))

        positions = telemetry.get("CarIdxPosition", [])
        if not isinstance(positions, (list, tuple)):
            return

        lap_dist_pcts = telemetry.get("CarIdxLapDistPct", [])
        laps = telemetry.get("CarIdxLap", [])
        best_lap_times = telemetry.get("CarIdxBestLapTime", [])
        last_lap_times = telemetry.get("CarIdxLastLapTime", [])
        on_pit_roads = telemetry.get("CarIdxOnPitRoad", [])
        p2p_statuses = telemetry.get("CarIdxP2P_Status", [])
        p2p_counts = telemetry.get("CarIdxP2P_Count", [])
        tire_compounds = telemetry.get("CarIdxTireCompound", [])
        track_surfaces = telemetry.get("CarIdxTrackSurface", [])
        rpms = telemetry.get("CarIdxRPM", [])
        gears = telemetry.get("CarIdxGear", [])

        # Build standings
        self.standings = []
        max_nearby = self.config.get("prompt", {}).get("include_nearby_cars", 5)

        for i in range(len(positions)):
            pos = positions[i] if i < len(positions) else 0
            if pos <= 0:
                continue

            name = self._driver_names.get(i, f"Car #{i}")

            self.standings.append(
                {
                    "car_idx": i,
                    "position": int(pos),
                    "lap_dist_pct": float(lap_dist_pcts[i])
                    if i < len(lap_dist_pcts)
                    else 0.0,
                    "lap": int(laps[i]) if i < len(laps) else 0,
                    "best_lap_time": float(best_lap_times[i])
                    if i < len(best_lap_times)
                    else -1.0,
                    "last_lap_time": float(last_lap_times[i])
                    if i < len(last_lap_times)
                    else -1.0,
                    "on_pit_road": bool(on_pit_roads[i])
                    if i < len(on_pit_roads)
                    else False,
                    "driver_name": name,
                }
            )

        # Sort by position
        self.standings.sort(key=lambda x: x["position"])

        # Build nearby cars — DriverState objects with shared-car fields populated
        nearby = []
        for entry in self.standings:
            idx = entry["car_idx"]
            if idx == player_idx:
                continue

            # Compute gap (rough — based on lap distance difference)
            other_lap_dist = entry["lap_dist_pct"]
            other_lap = entry["lap"]
            gap = (other_lap + other_lap_dist) - (self.player.lap + player_lap_dist_pct)

            if abs(entry["position"] - self.player.position) <= max_nearby:
                car = DriverState(
                    car_idx=idx,
                    driver_name=entry["driver_name"],
                    position=entry["position"],
                    lap=entry["lap"],
                    lap_dist_pct=other_lap_dist,
                    best_lap_time=entry["best_lap_time"],
                    last_lap_time=entry["last_lap_time"],
                    on_pit_road=entry["on_pit_road"],
                    p2p_remaining=int(p2p_counts[idx]) if idx < len(p2p_counts) else 0,
                    p2p_active=bool(p2p_statuses[idx])
                    if idx < len(p2p_statuses)
                    else False,
                    tire_compound=int(tire_compounds[idx])
                    if idx < len(tire_compounds)
                    else 0,
                    track_surface=int(track_surfaces[idx])
                    if idx < len(track_surfaces)
                    else 0,
                    rpm=float(rpms[idx]) if idx < len(rpms) else 0.0,
                    gear=int(gears[idx]) if idx < len(gears) else 0,
                    gap_seconds=gap,
                )
                nearby.append(car)

        self.nearby_cars = sorted(
            nearby, key=lambda c: abs(c.position - self.player.position)
        )[: max_nearby * 2]

    def _check_lap_completion(self, telemetry: dict):
        """Detect lap completions and record lap history for the player."""
        current_lap_completed = self.player.lap_completed

        if (
            current_lap_completed > self._last_lap_completed
            and self._last_lap_completed >= 0
        ):
            lap_time = self.player.last_lap_time
            if lap_time > 0:
                record = LapRecord(
                    lap_number=current_lap_completed,
                    lap_time=lap_time,
                    fuel_at_start=self._fuel_at_lap_start,
                    fuel_at_end=self.player.fuel_level,
                    fuel_used=self._fuel_at_lap_start - self.player.fuel_level
                    if self._fuel_at_lap_start > 0
                    else 0.0,
                    tyre_temps={
                        corner: TyreState(
                            temp_left=self.player.tyres[corner].temp_left,
                            temp_center=self.player.tyres[corner].temp_center,
                            temp_right=self.player.tyres[corner].temp_right,
                        )
                        for corner in ["LF", "RF", "LR", "RR"]
                        if corner in self.player.tyres
                    },
                    was_personal_best=lap_time
                    <= (
                        self.player.best_lap_time
                        if self.player.best_lap_time > 0
                        else lap_time
                    ),
                )

                # Keep only last N laps
                max_history = self.config.get("prompt", {}).get(
                    "include_lap_history", 10
                )
                self.player.lap_history.append(record)
                if len(self.player.lap_history) > max_history:
                    self.player.lap_history = self.player.lap_history[-max_history:]

        self._last_lap_completed = current_lap_completed
        self._fuel_at_lap_start = self.player.fuel_level

    # --- Derived values ---

    @property
    def fuel_laps_remaining(self) -> float:
        """Estimate how many laps of fuel remain for the player."""
        if not self.player.lap_history or self.player.fuel_use_per_hour <= 0:
            return 0.0

        recent = (
            self.player.lap_history[-5:]
            if len(self.player.lap_history) >= 5
            else self.player.lap_history
        )
        fuel_per_lap = sum(r.fuel_used for r in recent if r.fuel_used > 0)
        laps_with_fuel = sum(1 for r in recent if r.fuel_used > 0)

        if laps_with_fuel == 0:
            return 0.0

        avg_fuel_per_lap = fuel_per_lap / laps_with_fuel
        if avg_fuel_per_lap <= 0:
            return 0.0

        return self.player.fuel_level / avg_fuel_per_lap

    @property
    def lap_time_trend(self) -> float:
        """Seconds per lap degradation trend for the player (positive = getting slower)."""
        if len(self.player.lap_history) < 3:
            return 0.0
        recent = self.player.lap_history[-5:]
        times = [r.lap_time for r in recent if r.lap_time > 0]
        if len(times) < 2:
            return 0.0
        n = len(times)
        slope = (times[-1] - times[0]) / (n - 1) if n > 1 else 0.0
        return slope

    @property
    def flags_list(self) -> list[str]:
        """Human-readable list of active session flags."""
        flags = self.session.flags
        result = []
        if flags & FLAG_GREEN:
            result.append("Green")
        if flags & FLAG_YELLOW:
            result.append("Yellow")
        if flags & FLAG_RED:
            result.append("Red")
        if flags & FLAG_CHECKERED:
            result.append("Checkered")
        if flags & FLAG_WHITE:
            result.append("White (slow car)")
        if flags & FLAG_BLUE:
            result.append("Blue (faster car)")
        if flags & FLAG_RESTART:
            result.append("Restart")
        if not result:
            result.append("Green")  # Default to green if no flags set
        return result

    # --- Snapshot for context builder ---

    def get_snapshot(self) -> dict:
        """Return a condensed dict of the current race state for the LLM.

        This includes everything the context builder might need,
        filtered later by context_depth.
        """
        return {
            "session": {
                "track_name": self.session.track_name,
                "track_config": self.session.track_config,
                "session_type": self.session.session_type,
                "session_name": self.session.session_name,
                "laps_total": self.session.laps_total,
                "laps_remain": self.session.laps_remain,
                "time_remain": self.session.time_remain,
                "flags": self.flags_list,
                "race_laps": self.session.race_laps,
                "weather": {
                    "track_temp": self.session.track_temp,
                    "track_wetness": self.session.track_wetness,
                    "weather_declared_wet": self.session.weather_declared_wet,
                    "air_temp": self.session.air_temp,
                    "precipitation": self.session.precipitation,
                    "wind_dir": self.session.wind_dir,
                    "wind_vel": self.session.wind_vel,
                },
                "config": {
                    "fuel_max_litres": self.session.fuel_max_litres,
                    "is_fixed_setup": self.session.is_fixed_setup,
                    "incident_limit": self.session.incident_limit,
                    "fast_repairs_limit": self.session.fast_repairs_limit,
                    "track_num_turns": self.session.track_num_turns,
                    "track_length_km": self.session.track_length_km,
                    "pit_speed_limit_kph": self.session.pit_speed_limit_kph,
                    "num_starters": self.session.num_starters,
                    "idle_rpm": self.session.idle_rpm,
                    "redline_rpm": self.session.redline_rpm,
                    "shift_rpm": self.session.shift_rpm,
                },
            },
            "player": {
                "car_idx": self.player.car_idx,
                "driver_name": self.player.driver_name,
                "position": self.player.position,
                "class_position": self.player.class_position,
                "lap": self.player.lap,
                "lap_completed": self.player.lap_completed,
                "speed": self.player.speed,
                "is_on_track": self.player.is_on_track,
                "is_in_garage": self.player.is_in_garage,
                "fuel_level": self.player.fuel_level,
                "fuel_pct": self.player.fuel_pct,
                "fuel_use_per_hour": self.player.fuel_use_per_hour,
                "fuel_laps_remaining": self.fuel_laps_remaining,
                "current_lap_time": self.player.current_lap_time,
                "best_lap_time": self.player.best_lap_time,
                "last_lap_time": self.player.last_lap_time,
                "delta_to_best": self.player.delta_to_best,
                "lap_time_trend": self.lap_time_trend,
                # Engine health
                "oil_temp": self.player.oil_temp,
                "oil_press": self.player.oil_press,
                "oil_level": self.player.oil_level,
                "water_temp": self.player.water_temp,
                "water_level": self.player.water_level,
                "fuel_press": self.player.fuel_press,
                "engine_warnings": self.player.engine_warnings,
                "manifold_press": self.player.manifold_press,
                "voltage": self.player.voltage,
                # Damage and penalties
                "weight_penalty": self.player.weight_penalty,
                "fast_repairs_used": self.player.fast_repairs_used,
                "pit_repair_time_left": self.player.pit_repair_time_left,
                "pit_opt_repair_time_left": self.player.pit_opt_repair_time_left,
                # Proximity
                "car_left_right": self.player.car_left_right,
                "car_dist_ahead": self.player.car_dist_ahead,
                "car_dist_behind": self.player.car_dist_behind,
                "tow_time": self.player.tow_time,
                # G-forces
                "lat_accel": self.player.lat_accel,
                "long_accel": self.player.long_accel,
                "vert_accel": self.player.vert_accel,
                # Brake bias
                "brake_bias": self.player.brake_bias,
                # Shift lights
                "shift_indicator_pct": self.player.shift_indicator_pct,
                "shift_rpm": self.player.shift_rpm,
                # Tyres
                "tyres": {
                    corner: {
                        "temp_left": ts.temp_left,
                        "temp_center": ts.temp_center,
                        "temp_right": ts.temp_right,
                        "cold_pressure": ts.cold_pressure,
                        "wear_left": ts.wear_left,
                        "wear_center": ts.wear_center,
                        "wear_right": ts.wear_right,
                    }
                    for corner, ts in self.player.tyres.items()
                },
                "tyre_odometers": self.player.tyre_odometers,
                "brake_pressures": self.player.brake_pressures,
                "brake_abs_active": self.player.brake_abs_active,
                "incidents": self.player.incidents,
                "team_incidents": self.player.team_incidents,
                "on_pit_road": self.player.on_pit_road,
                "pitstop_active": self.player.pitstop_active,
                "pits_open": self.player.pits_open,
                "fast_repair_available": self.player.fast_repair_available,
                "p2p_active": self.player.p2p_active,
                "p2p_remaining": self.player.p2p_remaining,
                "tire_sets_available": self.player.tire_sets_available,
                "tire_sets_used": self.player.tire_sets_used,
                "tire_compound": self.player.tire_compound,
            },
            "nearby_cars": [
                {
                    "car_idx": car.car_idx,
                    "driver_name": car.driver_name,
                    "position": car.position,
                    "gap_seconds": car.gap_seconds,
                    "last_lap_time": car.last_lap_time,
                    "best_lap_time": car.best_lap_time,
                    "on_pit_road": car.on_pit_road,
                    "p2p_available": car.p2p_remaining > 0,
                    "p2p_remaining": car.p2p_remaining,
                    "tire_compound": car.tire_compound,
                    "track_surface": car.track_surface,
                    "rpm": car.rpm,
                    "gear": car.gear,
                }
                for car in self.nearby_cars
            ],
            "lap_history": [
                {
                    "lap_number": r.lap_number,
                    "lap_time": r.lap_time,
                    "fuel_used": r.fuel_used,
                    "was_personal_best": r.was_personal_best,
                }
                for r in self.player.lap_history
            ],
            "standings_count": len(self.standings),
            "tick_count": self._tick_count,
            "last_update_time": self._last_update_time,
        }
