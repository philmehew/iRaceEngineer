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

# --- Unit conversion helpers ---
# iRacing reports telemetry in various units (kPa, degF, etc.) depending
# on the car and session. These helpers convert from iRacing units to
# our display units, with sanity checks to catch wildly wrong values.

# kPa → PSI conversion factor
KPA_TO_PSI = 0.145038


def convert_kpa_to_psi(kpa: float) -> float:
    """Convert kPa to PSI."""
    return kpa * KPA_TO_PSI


def convert_f_to_c(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9


def sanitize_fuel_level(raw_fuel: float, fuel_pct: float, fuel_max: float) -> float:
    """Sanitize fuel level reading.

    iRacing's FuelLevel telemetry variable is supposed to be in litres,
    but some cars or sessions return values in different scales (e.g.
    millilitres, or percentage * 1000). If the raw value is wildly
    inconsistent with FuelLevelPct * tank_capacity, derive fuel from
    the percentage instead.

    Args:
        raw_fuel: The raw FuelLevel value from iRacing.
        fuel_pct: The FuelLevelPct value (0.0-1.0).
        fuel_max: The tank capacity in litres.

    Returns:
        Sanitised fuel level in litres.
    """
    if raw_fuel <= 0:
        return raw_fuel

    # If we have a tank capacity and the raw value exceeds it by 2x,
    # it's clearly wrong. Derive from percentage instead.
    if fuel_max > 0 and raw_fuel > fuel_max * 2:
        if 0 < fuel_pct <= 1.0:
            derived = fuel_pct * fuel_max
            logger.warning(
                f"FuelLevel {raw_fuel:.2f}L exceeds tank capacity "
                f"{fuel_max:.1f}L — deriving from FuelLevelPct "
                f"({fuel_pct:.3f} × {fuel_max:.1f}L = {derived:.2f}L)"
            )
            return derived
        # No valid percentage — can't derive, but flag it
        logger.warning(
            f"FuelLevel {raw_fuel:.2f}L exceeds tank capacity "
            f"{fuel_max:.1f}L and no valid FuelLevelPct ({fuel_pct:.3f})"
        )
        return raw_fuel

    # If raw fuel seems plausible but percentage contradicts it
    # (e.g. raw=7.5L but pct=0.09 suggests ~2L), use percentage
    if fuel_max > 0 and 0 < fuel_pct <= 1.0:
        derived = fuel_pct * fuel_max
        # If they disagree by more than 3x, trust the percentage
        if raw_fuel > 0 and derived > 0:
            ratio = raw_fuel / derived
            if ratio > 3 or ratio < 0.33:
                logger.warning(
                    f"FuelLevel {raw_fuel:.2f}L disagrees with "
                    f"FuelLevelPct ({fuel_pct:.3f} × {fuel_max:.1f}L = {derived:.2f}L) "
                    f"— using percentage-derived value"
                )
                return derived

    return raw_fuel


def sanitize_pressure(raw_pressure: float) -> float:
    """Sanitize tyre/brake pressure reading.

    iRacing reports pressures in kPa for most cars, but some cars or
    sessions may report in Pa, hPa, or other scaled units. This function
    normalizes values into the expected kPa range.

    Normal kPa ranges:
    - Tyre pressures: ~100-350 kPa (14-50 PSI)
    - Brake line pressures: ~300-20000 kPa

    Returns:
        Pressure in kPa (normalized to plausible range).
    """
    if raw_pressure <= 0:
        return raw_pressure

    # Normal kPa range — plausible as-is
    if 10 < raw_pressure <= 500:
        return raw_pressure

    # Very large values — try to normalize
    if raw_pressure > 100000:
        # Likely in Pascals — convert to kPa (divide by 1000)
        return raw_pressure / 1000

    if raw_pressure > 5000:
        # Could be in hectopascals (÷100) or 10x kPa (÷10).
        # Try ÷100 first (gives plausible tyre pressures for MX-5 style values)
        # e.g. 6600 ÷ 100 = 66 kPa (9.6 PSI) — plausible for cold tyres
        # e.g. 11000 ÷ 100 = 110 kPa (16 PSI) — plausible
        # e.g. 18500 ÷ 100 = 185 kPa (26.8 PSI) — plausible for hot tyres
        candidate = raw_pressure / 100
        if 10 < candidate <= 500:
            return candidate
        # Try ÷10 if ÷100 gives too-low values
        # e.g. 6600 ÷ 10 = 660 kPa (95.7 PSI) — too high for tyres
        # This probably won't help, but try anyway
        candidate = raw_pressure / 10
        if 10 < candidate <= 500:
            return candidate
        # Nothing works — return as-is and let display handle it
        return raw_pressure

    if raw_pressure > 500:
        # 500-5000: could be in Pa (÷1000 gives 0.5-5, too low for tyres)
        # or 10x kPa (÷10 gives 50-500, plausible for brake pressures)
        # For tyre pressures, this range is too high.
        # Try ÷10 first
        candidate = raw_pressure / 10
        if 10 < candidate <= 500:
            return candidate
        return raw_pressure

    # 0-10 range: very low kPa. Could be in bar (1-3 bar = 14-43 PSI)
    # but too ambiguous to convert. Return as-is.
    return raw_pressure


@dataclass
class EngineSnapshot:
    """Engine health readings captured at a point in time."""

    oil_temp: float = 0.0
    oil_press: float = 0.0
    water_temp: float = 0.0
    voltage: float = 0.0
    manifold_press: float = 0.0


@dataclass
class TyreState:
    """Tyre data for one corner."""

    temp_left: float = 0.0  # Inner/cold side temp (°C)
    temp_center: float = 0.0  # Center temp (°C)
    temp_right: float = 0.0  # Outer/hot side temp (°C)
    cold_pressure: float = 0.0  # kPa (iRacing unit; converted to PSI at display)
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
    engine: EngineSnapshot = field(default_factory=EngineSnapshot)
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
    player_track_surface: int = 0  # iRacing PlayerTrackSurface: -1=not in world, 0=garage, 1=pit stall, 2=pit road, 3=on track

    # Player-only: proximity
    car_left_right: int = 0  # 0=none, 1=car left, 2=car right, 3=both
    car_dist_ahead: float = 0.0  # metres (iRacing CarDistAhead)
    car_dist_behind: float = 0.0  # metres (iRacing CarDistBehind)
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
    time_remain: float = 0.0  # Seconds remaining in session
    session_time: float = 0.0  # Total session duration in seconds (e.g. 1800 = 30 min)
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
    fuel_max_litres: float = 0.0  # Tank capacity (full)
    fuel_max_pct: float = (
        1.0  # Max fuel load as fraction of tank (e.g. 0.4 for 40% restriction)
    )
    fuel_max_start_litres: float = 0.0  # Derived: max fuel at start = tank × max_pct
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
    est_lap_time: float = 0.0  # DriverCarEstLapTime from session info


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
        self._last_lap: int = -1  # Track Lap counter to detect new laps
        self._fuel_at_lap_start: float = 0.0
        self._was_on_pit_road: bool = False  # Track pit road transitions

        # Tyre staleness tracking: store previous tick's tyre values
        self._prev_tyre_values: dict[str, TyreState] | None = None

        # Teammate real names: car_idx → config name (for showing both names)
        self._teammate_real_names: dict[int, str] = {}

        # Driver name aliases: iRacing UserName → real name (from config)
        self._driver_aliases: dict[str, str] = self.config.get("driver_aliases", {})

        # Timing
        self._last_update_time: float = 0.0
        self._tick_count: int = 0

        # Telemetry units (populated from iRacing VarHeader.unit each tick)
        self._telemetry_units: dict[str, str] = {}

        # Driver name lookup (set by main from iracing_client)
        self._driver_names: dict[int, str] = {}

        # Team car indices (populated when session info is available)
        self._team_car_indices: set[int] = set()

    def set_driver_names(self, names: dict[int, str]):
        """Set driver name lookup from IRacingClient."""
        self._driver_names = names

    def set_team_indices(
        self, indices: set[int], driver_aliases: dict[str, str] | None = None
    ):
        """Set which car indices are team drivers (for full tracking).

        Also resolves teammate real names using driver_aliases.
        driver_aliases maps iRacing UserName → real name (e.g. "Evolution" → "Patrik Farsang").
        """
        self._team_car_indices = indices

        # Merge provided aliases with any in config
        aliases = dict(self.config.get("driver_aliases", {}))
        if driver_aliases:
            aliases.update(driver_aliases)
        self._driver_aliases = aliases

        # Build mapping of teammate car_idx → real name
        self._teammate_real_names = {}
        for idx in indices:
            display_name = self._driver_names.get(idx, "")
            if display_name in aliases and aliases[display_name] != display_name:
                self._teammate_real_names[idx] = aliases[display_name]
            else:
                self._teammate_real_names[idx] = ""

    def update(
        self,
        telemetry: dict,
        session_info: dict,
        driver_names: dict[int, str] | None = None,
        units: dict[str, str] | None = None,
    ):
        """Update race state from telemetry data and session info.

        Called each tick from the main poll loop.

        Args:
            telemetry: Dict of telemetry variable name -> raw value.
            session_info: Parsed session info YAML data.
            driver_names: Optional mapping of car_idx -> driver name.
            units: Optional dict of telemetry variable name -> iRacing unit
                string (e.g. 'kPa', 'degC', 'm'). Used for unit conversion.
        """
        if driver_names:
            self._driver_names = driver_names

        # Cache telemetry units for use by _update_player and others
        if units:
            self._telemetry_units = units

        self._tick_count += 1
        self._last_update_time = time.time()

        # Detect iRacing session reset: Lap drops to 0 after we've been
        # racing (e.g. session ends, new session loads). When this happens,
        # preserve the last known state instead of overwriting with zeros.
        new_lap = int(telemetry.get("Lap", 0))
        if new_lap == 0 and self._last_lap > 0:
            # Session ended / reset — keep last known state, don't process
            # the zeroed-out telemetry that iRacing sends after the chequered.
            return

        # Update session state
        self._update_session(telemetry, session_info)

        # Update player state (pass units for conversion)
        self._update_player(telemetry, units=self._telemetry_units)

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
                    # Parse total session time (e.g. "1800.0000 sec" or float seconds)
                    session_time_raw = s.get("SessionTime", 0)
                    if isinstance(session_time_raw, str):
                        # "1800.0000 sec" -> 1800.0
                        session_time_raw = session_time_raw.replace(" sec", "").strip()
                    try:
                        self.session.session_time = float(session_time_raw or 0)
                    except (ValueError, TypeError):
                        self.session.session_time = 0.0
                    break

            # Driver info — car spec and shift RPMs
            driver_info = session_info.get("DriverInfo", {})
            self.session.fuel_max_litres = float(
                driver_info.get("DriverCarFuelMaxLtr", 0.0)
            )
            # DriverCarMaxFuelPct is the max fuel load as a fraction of the
            # full tank (e.g. 0.4 means only 40% of the tank can be used).
            # Default 1.0 = no restriction.
            self.session.fuel_max_pct = float(
                driver_info.get("DriverCarMaxFuelPct", 1.0)
            )
            if self.session.fuel_max_pct <= 0:
                self.session.fuel_max_pct = 1.0  # Safety default
            # Derived: the actual max fuel at session start
            self.session.fuel_max_start_litres = (
                self.session.fuel_max_litres * self.session.fuel_max_pct
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
            self.session.est_lap_time = float(
                driver_info.get("DriverCarEstLapTime", 0.0)
            )

    def _update_player(self, telemetry: dict, units: dict | None = None):
        """Update player car state from telemetry.

        Args:
            telemetry: Dict of telemetry variable name -> raw value.
            units: Optional dict of telemetry variable name -> iRacing unit
                string (e.g. 'kPa', 'degC', 'm'). Used for unit conversion.
        """
        p = self.player
        if units is None:
            units = {}

        # Save previous tyre values for staleness detection
        if p.tyres:
            self._prev_tyre_values = {
                corner: TyreState(
                    temp_left=ts.temp_left,
                    temp_center=ts.temp_center,
                    temp_right=ts.temp_right,
                    cold_pressure=ts.cold_pressure,
                    wear_left=ts.wear_left,
                    wear_center=ts.wear_center,
                    wear_right=ts.wear_right,
                )
                for corner, ts in p.tyres.items()
            }

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

        # Engine health (player only) — convert units where needed
        raw_oil_temp = float(telemetry.get("OilTemp", 0.0))
        raw_water_temp = float(telemetry.get("WaterTemp", 0.0))
        # iRacing can report temps in degF depending on car/settings
        if units.get("OilTemp") == "degF" and raw_oil_temp > 0:
            raw_oil_temp = convert_f_to_c(raw_oil_temp)
        if units.get("WaterTemp") == "degF" and raw_water_temp > 0:
            raw_water_temp = convert_f_to_c(raw_water_temp)
        p.oil_temp = raw_oil_temp
        p.oil_press = float(telemetry.get("OilPress", 0.0))
        p.oil_level = float(telemetry.get("OilLevel", 0.0))
        p.water_temp = raw_water_temp
        p.water_level = float(telemetry.get("WaterLevel", 0.0))
        p.fuel_press = float(telemetry.get("FuelPress", 0.0))
        p.engine_warnings = int(telemetry.get("EngineWarnings", 0))
        p.manifold_press = float(telemetry.get("ManifoldPress", 0.0))

        # Voltage — sanity check: car voltage should be 10-18V
        raw_voltage = float(telemetry.get("Voltage", 0.0))
        if raw_voltage > 50:
            # Likely in wrong unit — divide by 5 as a rough heuristic
            # (iRacing sometimes reports Voltage * 5 for some cars)
            p.voltage = raw_voltage / 5
            logger.debug(
                f"Voltage {raw_voltage:.1f} exceeds 50V — "
                f"assuming 5x scaling, using {p.voltage:.1f}V"
            )
        else:
            p.voltage = raw_voltage

        # Car status
        p.is_on_track = bool(telemetry.get("IsOnTrack", False))
        p.is_in_garage = bool(telemetry.get("IsInGarage", False))
        p.player_track_surface = int(telemetry.get("PlayerTrackSurface", 0))

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
        # CarDistAhead/Behind are in metres per iRacing SDK, not seconds.
        # We store them as-is for now; context_builder labels them correctly.
        p.car_left_right = int(telemetry.get("CarLeftRight", 0))
        p.car_dist_ahead = float(telemetry.get("CarDistAhead", 0.0))
        p.car_dist_behind = float(telemetry.get("CarDistBehind", 0.0))
        p.tow_time = float(telemetry.get("PlayerCarTowTime", 0.0))

        # Shift lights (player only)
        p.shift_indicator_pct = float(telemetry.get("ShiftIndicatorPct", 0.0))
        p.shift_rpm = float(telemetry.get("PlayerCarSLShiftRPM", 0.0))

        # Brake bias (player only) — sanity check: should be 0-100%
        raw_brake_bias = float(telemetry.get("dcBrakeBias", 0.0))
        # dcBrakeBias in iRacing is typically 0.0-1.0 (fraction), but some
        # cars report it differently. If >1, it's likely a percentage already
        # or in a different scale. Values >200 are clearly wrong.
        if raw_brake_bias > 200:
            p.brake_bias = 0.0  # clearly invalid data
            logger.debug(f"Brake bias {raw_brake_bias:.1f} exceeds 200% — ignoring")
        elif raw_brake_bias > 1.0:
            # Likely already a percentage (e.g. 54.5 means 54.5%)
            p.brake_bias = raw_brake_bias
        else:
            # Fraction (0.0-1.0), convert to percentage
            p.brake_bias = raw_brake_bias * 100

        # Fuel (player only) — sanitise fuel level against tank capacity
        raw_fuel_level = float(telemetry.get("FuelLevel", 0.0))
        p.fuel_pct = float(telemetry.get("FuelLevelPct", 0.0))
        # Clamp fuel_pct to 0-1 range (iRacing sometimes reports >1 briefly)
        if p.fuel_pct > 1.5:
            # Likely a percentage (0-100) rather than fraction (0-1)
            p.fuel_pct = p.fuel_pct / 100
        p.fuel_pct = max(0.0, min(p.fuel_pct, 1.0))
        # Use the effective max fuel (tank × restriction) for validation,
        # since fuel_pct is relative to the restricted amount, not the full tank.
        effective_max = (
            self.session.fuel_max_start_litres
            if self.session.fuel_max_start_litres > 0
            else self.session.fuel_max_litres
        )
        p.fuel_level = sanitize_fuel_level(raw_fuel_level, p.fuel_pct, effective_max)
        p.fuel_use_per_hour = float(telemetry.get("FuelUsePerHour", 0.0))

        # Tyres (per corner — player only for temps/wear)
        # iRacing reports pressures in kPa; store raw kPa and convert at display time
        for corner in ["LF", "RF", "LR", "RR"]:
            prefix = corner
            # Temperature conversion: some cars report in degF
            raw_temp_l = float(telemetry.get(f"{prefix}tempCL", 0.0))
            raw_temp_c = float(telemetry.get(f"{prefix}tempCM", 0.0))
            raw_temp_r = float(telemetry.get(f"{prefix}tempCR", 0.0))
            temp_unit = units.get(f"{prefix}tempCL", "")
            if temp_unit == "degF":
                raw_temp_l = convert_f_to_c(raw_temp_l) if raw_temp_l > 0 else 0.0
                raw_temp_c = convert_f_to_c(raw_temp_c) if raw_temp_c > 0 else 0.0
                raw_temp_r = convert_f_to_c(raw_temp_r) if raw_temp_r > 0 else 0.0
            # Pressure: iRacing reports in kPa. Sanitize wildly wrong values.
            raw_pressure = float(telemetry.get(f"{prefix}coldPressure", 0.0))
            raw_pressure = sanitize_pressure(raw_pressure)
            p.tyres[corner] = TyreState(
                temp_left=raw_temp_l,
                temp_center=raw_temp_c,
                temp_right=raw_temp_r,
                cold_pressure=raw_pressure,  # stored in kPa
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

        # Brakes (player only) — pressures in kPa from iRacing
        p.brake_pressures = {
            "LF": sanitize_pressure(float(telemetry.get("LFbrakeLinePress", 0.0))),
            "RF": sanitize_pressure(float(telemetry.get("RFbrakeLinePress", 0.0))),
            "LR": sanitize_pressure(float(telemetry.get("LRbrakeLinePress", 0.0))),
            "RR": sanitize_pressure(float(telemetry.get("RRbrakeLinePress", 0.0))),
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
        """Detect lap completions and record lap history for the player.

        Fuel tracking: _fuel_at_lap_start is captured once when a new lap
        begins (Lap counter increments) and held until that lap completes.
        This gives accurate per-lap fuel consumption instead of the tiny
        tick-to-tick delta that the old code produced (~0.01L instead of ~1.3L).
        """
        current_lap_completed = self.player.lap_completed
        current_lap = self.player.lap

        # Record the completed lap FIRST (using the old _fuel_at_lap_start
        # from when this lap began), before we update it for the new lap.
        if (
            current_lap_completed > self._last_lap_completed
            and self._last_lap_completed >= 0
        ):
            lap_time = self.player.last_lap_time
            # iRacing sometimes clears LapLastLapTime to 0 at the instant
            # LapCompleted increments. Fall back to CarIdxLastLapTime.
            if lap_time <= 0 and telemetry:
                idx_lap_times = telemetry.get("CarIdxLastLapTime", [])
                car_idx = self.player.car_idx
                if (
                    car_idx >= 0
                    and car_idx < len(idx_lap_times)
                    and idx_lap_times[car_idx] > 0
                ):
                    lap_time = float(idx_lap_times[car_idx])
            if lap_time > 0:
                # Fuel used this lap: start - end. Clamp to 0 — negative
                # means the car refuelled (pit stop), which isn't "fuel saved".
                fuel_used = max(
                    0.0,
                    self._fuel_at_lap_start - self.player.fuel_level
                    if self._fuel_at_lap_start > 0
                    else 0.0,
                )
                # Sanity check: fuel used per lap should be 0.1-15L for any
                # realistic race car. Wildly wrong telemetry can produce
                # values like 4275L/lap which corrupt all downstream
                # calculations (fuel range, shortage warnings, etc.).
                if fuel_used > 15:
                    logger.warning(
                        f"Fuel used {fuel_used:.1f}L for lap "
                        f"{current_lap_completed} exceeds 15L sanity "
                        f"threshold — discarding (start="
                        f"{self._fuel_at_lap_start:.2f}, end="
                        f"{self.player.fuel_level:.2f})"
                    )
                    fuel_used = 0.0
                record = LapRecord(
                    lap_number=current_lap_completed,
                    lap_time=lap_time,
                    fuel_at_start=self._fuel_at_lap_start,
                    fuel_at_end=self.player.fuel_level,
                    fuel_used=fuel_used,
                    tyre_temps={
                        corner: TyreState(
                            temp_left=self.player.tyres[corner].temp_left,
                            temp_center=self.player.tyres[corner].temp_center,
                            temp_right=self.player.tyres[corner].temp_right,
                        )
                        for corner in ["LF", "RF", "LR", "RR"]
                        if corner in self.player.tyres
                    },
                    engine=EngineSnapshot(
                        oil_temp=self.player.oil_temp,
                        oil_press=self.player.oil_press,
                        water_temp=self.player.water_temp,
                        voltage=self.player.voltage,
                        manifold_press=self.player.manifold_press,
                    ),
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

        # After recording the lap, capture fuel for the START of the new lap.
        # When Lap increments (new lap begins), the current fuel_level is the
        # fuel at the start of this new lap (= fuel at end of previous lap).
        if current_lap > self._last_lap and self._last_lap >= 0:
            self._fuel_at_lap_start = self.player.fuel_level

        # Detect pit exit: when the car leaves pit road, reset the fuel
        # baseline so the next lap's fuel_used calculation isn't skewed by
        # the refuel (e.g. fuel going from 0.22L to 1.73L mid-lap).
        if self._was_on_pit_road and not self.player.on_pit_road:
            self._fuel_at_lap_start = self.player.fuel_level
        self._was_on_pit_road = self.player.on_pit_road

        self._last_lap_completed = current_lap_completed
        self._last_lap = current_lap

    # --- Derived values ---

    @property
    def avg_fuel_per_lap(self) -> float:
        """Rolling average fuel consumption per lap (last 3 completed laps).

        Returns 0.0 if no lap history with fuel data is available.
        """
        if not self.player.lap_history:
            return 0.0

        recent = self.player.lap_history[-3:]
        fuel_per_lap = sum(r.fuel_used for r in recent if r.fuel_used > 0)
        laps_with_fuel = sum(1 for r in recent if r.fuel_used > 0)

        if laps_with_fuel > 0 and fuel_per_lap > 0:
            return fuel_per_lap / laps_with_fuel
        return 0.0

    @property
    def fuel_est_quality(self) -> str:
        """Quality indicator for fuel laps remaining estimate.

        Returns:
            'good' — ≥3 laps of history, reliable estimate
            'rough' — 1-2 laps of history, approximate
            'unreliable' — no lap history, based on instantaneous rate
        """
        laps_with_fuel = sum(1 for r in self.player.lap_history if r.fuel_used > 0)
        if laps_with_fuel >= 3:
            return "good"
        elif laps_with_fuel >= 1:
            return "rough"
        return "unreliable"

    @property
    def engine_baseline(self) -> EngineSnapshot | None:
        """Auto-calibrated engine baseline from the first N completed laps.

        Returns averaged engine values from the calibration period, or None
        if not enough laps have been completed yet. Used by the context
        builder to flag values that deviate from early-race norms.

        N is configured via config.state.calibration_laps (default 5).
        """
        if not self.player.lap_history:
            return None

        cal_laps = self.config.get("state", {}).get("calibration_laps", 5)
        calibration = self.player.lap_history[:cal_laps]

        # Only use laps that have non-zero engine data
        valid = [r for r in calibration if r.engine.oil_temp > 0]
        if len(valid) < 2:  # Need at least 2 laps for a meaningful average
            return None

        n = len(valid)
        return EngineSnapshot(
            oil_temp=sum(r.engine.oil_temp for r in valid) / n,
            oil_press=sum(r.engine.oil_press for r in valid) / n,
            water_temp=sum(r.engine.water_temp for r in valid) / n,
            voltage=sum(r.engine.voltage for r in valid) / n,
            manifold_press=sum(r.engine.manifold_press for r in valid) / n,
        )

    @property
    def tyre_baseline(self) -> dict[str, float] | None:
        """Auto-calibrated tyre temp baseline from the first N completed laps.

        Returns {corner: avg_center_temp} from the calibration period, or None
        if not enough laps have been completed yet.

        N is configured via config.state.calibration_laps (default 5).
        """
        if not self.player.lap_history:
            return None

        cal_laps = self.config.get("state", {}).get("calibration_laps", 5)
        calibration = self.player.lap_history[:cal_laps]

        # Collect per-corner temps from laps that have tyre data
        corner_sums: dict[str, float] = {}
        corner_counts: dict[str, int] = {}
        for r in calibration:
            for corner, ts in r.tyre_temps.items():
                if ts.temp_center > 0:
                    corner_sums[corner] = corner_sums.get(corner, 0) + ts.temp_center
                    corner_counts[corner] = corner_counts.get(corner, 0) + 1

        # Need at least 2 laps with data for each corner
        if not corner_counts or min(corner_counts.values()) < 2:
            return None

        return {c: corner_sums[c] / corner_counts[c] for c in corner_sums}

    @property
    def fuel_laps_remaining(self) -> float:
        """Estimate how many laps of fuel remain for the player.

        Uses lap history when available. Falls back to burn rate + estimated
        lap time when no history exists yet (e.g. early race or mid-race restart).
        The fallback is unreliable — fuel_est_quality will be "unreliable" and
        the context builder will hide the number.
        """
        # Primary method: use per-lap average from lap history
        avg = self.avg_fuel_per_lap
        if avg > 0:
            return self.player.fuel_level / avg

        # Fallback: estimate from instantaneous burn rate and lap time.
        # This is wildly unreliable (varies with throttle position) — the
        # context builder will not show the resulting number to the LLM.
        fuel_rate = self.player.fuel_use_per_hour  # L/hr
        if fuel_rate <= 0:
            return 0.0

        # Use best lap time if available, then current lap time, then
        # session estimated lap time from DriverInfo
        est_lap_time = self.player.best_lap_time
        if est_lap_time <= 0:
            est_lap_time = self.player.last_lap_time
        if est_lap_time <= 0:
            est_lap_time = self.player.current_lap_time
        if est_lap_time <= 0:
            # Last resort: use DriverCarEstLapTime from session
            est_lap_time = getattr(self.session, "est_lap_time", 0) or 0
        if est_lap_time <= 0:
            return 0.0  # No way to estimate without any time reference

        # fuel_per_lap = fuel_rate * (lap_time_seconds / 3600)
        fuel_per_lap = fuel_rate * (est_lap_time / 3600.0)
        if fuel_per_lap <= 0:
            return 0.0

        return self.player.fuel_level / fuel_per_lap

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
    def tyre_staleness(self) -> str:
        """Determine if tyre data is stale (frozen on track) or live.

        Uses two signals:
        1. Primary: whether the car is on track or in pits. Most iRacing cars
           freeze tyre data (temps, pressures, wear) while on track — it only
           updates during pit stops. So on-track data is presumed stale unless
           proven otherwise.
        2. Secondary: tick-to-tick comparison. If tyre values are actively
           changing between updates, they're definitely live regardless of
           car location.

        Returns:
            'live' — tyre values are actively changing (real-time data)
            'stale' — car is on track and data is likely frozen
            'unknown' — not enough info (car off track, no data, or first tick)
        """
        # If car is in pits, tyre data is likely live (just updated)
        if self.player.on_pit_road:
            return "live"

        # If car is off track entirely, we can't determine staleness
        if not self.player.is_on_track:
            return "unknown"

        # Car is on track — tyre data is likely frozen for most iRacing cars.
        # But check tick-to-tick: if values ARE actively changing, they're live.
        if hasattr(self, "_prev_tyre_values") and self._prev_tyre_values:
            curr = self.player.tyres
            prev = self._prev_tyre_values
            for corner in ["LF", "RF", "LR", "RR"]:
                if corner not in curr or corner not in prev:
                    continue
                c = curr[corner]
                p = prev[corner]
                if abs(c.temp_center - p.temp_center) > 0.05:
                    return "live"
                if abs(c.cold_pressure - p.cold_pressure) > 0.01:
                    return "live"
                if abs(c.wear_center - p.wear_center) > 0.0001:
                    return "live"

        # On track and values aren't changing (or no previous tick to compare)
        # → presume stale (frozen data, iRacing limitation for most cars)
        return "stale"

    @property
    def estimated_total_laps(self) -> int | None:
        """Estimate total race laps from session time and average lap time.

        For time-based races (SessionLapsRemain >= 32767), estimate total laps
        from remaining session time divided by average lap time.

        For lap-based races, use the actual total.

        Returns None if estimation is not possible.
        """
        LAPS_REMAIN_SENTINEL = 32767  # iRacing sentinel for unlimited/time-based

        laps_remain = self.session.laps_remain
        race_laps = self.session.race_laps

        # If this is a lap-based race, we can compute total directly
        if laps_remain > 0 and laps_remain < LAPS_REMAIN_SENTINEL:
            return race_laps + laps_remain

        # Time-based race: estimate from remaining session time
        time_remain = self.session.time_remain
        if time_remain <= 0:
            return None

        # Get the best available lap time for estimation
        avg_lap = self.player.best_lap_time
        if avg_lap <= 0:
            avg_lap = self.player.last_lap_time
        if avg_lap <= 0 and self.player.lap_history:
            # Use average of recent lap times
            recent = self.player.lap_history[-5:]
            times = [r.lap_time for r in recent if r.lap_time > 0]
            if times:
                avg_lap = sum(times) / len(times)
        if avg_lap <= 0:
            # Last resort: use DriverCarEstLapTime from session
            avg_lap = getattr(self.session, "est_lap_time", 0) or 0
        if avg_lap <= 0:
            return None

        # estimated laps remaining = time_remain / avg_lap_time
        laps_estimated = time_remain / avg_lap
        return race_laps + int(laps_estimated)

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
                "session_time": self.session.session_time,
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
                    "fuel_max_pct": self.session.fuel_max_pct,
                    "fuel_max_start_litres": self.session.fuel_max_start_litres,
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
                    "est_lap_time": self.session.est_lap_time,
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
                "player_track_surface": self.player.player_track_surface,
                "fuel_level": self.player.fuel_level,
                "fuel_pct": self.player.fuel_pct,
                "fuel_use_per_hour": self.player.fuel_use_per_hour,
                "fuel_laps_remaining": self.fuel_laps_remaining,
                "fuel_est_quality": self.fuel_est_quality,
                "avg_fuel_per_lap": self.avg_fuel_per_lap,
                "tyre_staleness": self.tyre_staleness,
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
                # Baselines (auto-calibrated from first N laps)
                "engine_baseline": (
                    {
                        "oil_temp": self.engine_baseline.oil_temp,
                        "oil_press": self.engine_baseline.oil_press,
                        "water_temp": self.engine_baseline.water_temp,
                        "voltage": self.engine_baseline.voltage,
                        "manifold_press": self.engine_baseline.manifold_press,
                    }
                    if self.engine_baseline
                    else None
                ),
                "tyre_baseline": self.tyre_baseline,
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
                    "is_teammate": car.car_idx in self._team_car_indices,
                    "real_name": self._teammate_real_names.get(car.car_idx, ""),
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
            "estimated_total_laps": self.estimated_total_laps,
            "driver_aliases": self._driver_aliases,
        }
