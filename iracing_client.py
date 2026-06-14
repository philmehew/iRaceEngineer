"""
iRacing data layer — reads shared memory via pyirsdk and exposes
structured telemetry, session info, and command methods.
"""

import logging
from dataclasses import dataclass
from typing import Any

import irsdk

logger = logging.getLogger(__name__)


# --- Data classes for structured output ---


@dataclass
class DriverInfo:
    """Per-driver metadata from session info."""

    car_idx: int
    driver_name: str = ""
    team_name: str = ""
    car_number: str = ""
    car_name: str = ""
    car_class: str = ""
    car_class_id: int = 0
    irating: int = 0
    lic_string: str = ""


@dataclass
class DriverState:
    """Per-driver current state from telemetry."""

    car_idx: int
    position: int = 0
    class_position: int = 0
    lap: int = 0
    lap_completed: int = 0
    lap_dist_pct: float = 0.0
    on_pit_road: bool = False
    track_surface: int = 0  # CarIdxTrackSurface: -1=not in world, 0=garage, 1=approaching, 2=on track (rare), 3=on track
    best_lap_time: float = -1.0
    last_lap_time: float = -1.0
    speed: float = 0.0
    rpm: float = 0.0
    gear: int = 0
    p2p_status: int = 0
    p2p_count: int = 0
    tire_compound: int = 0
    incidents: int = 0


@dataclass
class SessionInfo:
    """Session-level metadata."""

    track_name: str = ""
    track_config: str = ""
    session_type: str = ""
    session_num: int = 0
    session_name: str = ""
    laps_total: int = 0
    laps_remain: int = 0
    time_remain: float = 0.0
    flags: int = 0  # SessionFlags bitmask (session-wide)
    player_flags: int = (
        0  # CarIdxSessionFlags for player car (driver-specific: black, repair, etc.)
    )
    session_state: int = 0


class IRacingClient:
    """Wraps pyirsdk to provide structured data access and command methods."""

    def __init__(self):
        self._ir = irsdk.IRSDK()
        self._connected = False
        self._session_info_cache: dict[str, Any] | None = None
        self._session_info_tick: int = -1
        self._drivers: list[DriverInfo] = []

    # --- Connection ---

    def startup(self, test_file: str | None = None) -> bool:
        """Connect to iRacing shared memory.

        Args:
            test_file: Optional path to a pyirsdk dump file for testing.

        Returns:
            True if successfully connected and data is valid.
        """
        try:
            self._ir.startup(test_file=test_file)
            self._connected = self._ir.is_connected
            if self._connected:
                logger.info("Connected to iRacing")
                self._refresh_session_info()
            else:
                logger.warning("iRacing not running or no session active")
            return self._connected
        except Exception as e:
            logger.error(f"Failed to connect to iRacing: {e}")
            self._connected = False
            return False

    def shutdown(self):
        """Disconnect from iRacing shared memory."""
        self._ir.shutdown()
        self._connected = False
        logger.info("Disconnected from iRacing")

    @property
    def is_connected(self) -> bool:
        """Check if iRacing is still connected and data is fresh."""
        if not self._connected:
            return False
        # pyirsdk updates connection state on each data access
        # but we check explicitly
        try:
            self._connected = self._ir.is_connected
        except Exception:
            self._connected = False
        return self._connected

    # --- Telemetry (tick-level, fast) ---

    def get_telemetry(self) -> dict[str, Any]:
        """Read all telemetry variables from shared memory.

        Returns a flat dict of variable name -> value for all available
        telemetry fields. This captures everything so downstream consumers
        can pick what they need.
        """
        if not self.is_connected:
            return {}

        result: dict[str, Any] = {}
        for name in self._ir.var_headers_names:
            try:
                result[name] = self._ir[name]
            except Exception:
                # Some variables may not be available in all sessions
                pass
        return result

    def get_telemetry_value(self, name: str, default: Any = None) -> Any:
        """Read a single telemetry variable by name."""
        try:
            val = self._ir[name]
            return val if val is not None else default
        except (KeyError, TypeError):
            return default

    # --- Session info (slower, cached) ---

    def get_session_info(self) -> dict[str, Any]:
        """Get parsed session info YAML data.

        This is cached and only refreshed when the session info updates.
        """
        if not self.is_connected:
            return {}

        # Check if session info has been updated
        current_tick = getattr(self._ir, "session_info_update", -1)
        if current_tick != self._session_info_tick:
            self._refresh_session_info()

        return self._session_info_cache or {}

    def _refresh_session_info(self):
        """Parse and cache session info from iRacing.

        Fetches individual YAML sections via ir[key] (which calls
        _get_session_info internally). The sections we need are:
        WeekendInfo, SessionInfo, DriverInfo.
        """
        try:
            info: dict[str, Any] = {}
            for key in ("WeekendInfo", "SessionInfo", "DriverInfo", "SplitTimeInfo"):
                section = self._ir[key]
                if section:
                    info[key] = section
                elif key == "SplitTimeInfo":
                    logger.debug(
                        "SplitTimeInfo not available from iRacing session YAML "
                        "— sector times will not be tracked"
                    )
            self._session_info_cache = info
            self._session_info_tick = getattr(self._ir, "session_info_update", -1)
            self._parse_drivers()
        except Exception as e:
            logger.warning(f"Failed to refresh session info: {e}")

    def _parse_drivers(self):
        """Extract driver info from session info."""
        self._drivers = []
        if not self._session_info_cache:
            return

        driver_info = self._session_info_cache.get("DriverInfo", {})
        drivers = driver_info.get("Drivers", [])

        for d in drivers:
            self._drivers.append(
                DriverInfo(
                    car_idx=int(d.get("CarIdx", 0)),
                    driver_name=d.get("UserName", ""),
                    team_name=d.get("TeamName", ""),
                    car_number=d.get("CarNumberRaw", d.get("CarNumber", "")),
                    car_name=d.get("CarPath", "").split("/")[-1]
                    if d.get("CarPath")
                    else "",
                    car_class=d.get("CarClassShortName", ""),
                    car_class_id=int(d.get("CarClassId", 0)),
                    irating=int(d.get("IRating", 0)),
                    lic_string=d.get("LicString", ""),
                )
            )

    # --- Team detection ---

    def detect_team_indices(self, config: dict | None = None) -> set[int]:
        """Detect teammate car indices using TeamName and/or config fallback.

        Strategy:
        1. If auto_detect is enabled, find all drivers with the same TeamName
           as the player.
        2. If the config has an explicit teammates list (usernames or car numbers),
           match those too.
        3. Returns a set of car_idx values for all identified teammates
           (including the player).
        """
        team_indices: set[int] = set()
        team_config = (config or {}).get("team", {})
        auto_detect = team_config.get("auto_detect", True)

        # Get player's car index (graceful if not connected)
        player_idx = -1
        try:
            player_idx = self.get_telemetry_value("PlayerCarIdx", -1)
        except Exception:
            pass

        if player_idx is not None and player_idx >= 0:
            team_indices.add(player_idx)

        # Strategy 1: Auto-detect by TeamName
        if auto_detect and self._drivers:
            # Find the player's TeamName
            player_team = ""
            for d in self._drivers:
                if d.car_idx == player_idx:
                    player_team = d.team_name
                    break

            if player_team:
                # All drivers with the same team name are teammates
                for d in self._drivers:
                    if d.team_name == player_team and d.team_name:
                        team_indices.add(d.car_idx)
                        logger.debug(
                            f"Auto-detected teammate: {d.driver_name} (#{d.car_number}) via TeamName '{player_team}'"
                        )

        # Strategy 2: Config fallback — match by username or car number
        # Teammates list can contain:
        #   - Plain strings: "Wayne Smith8" (matches iRacing UserName, displayed as-is)
        #   - Mappings: "Evolution: Patrik Farsang" (nickname -> real name for display)
        explicit_teammates = team_config.get("teammates", [])
        explicit_car_numbers = team_config.get("car_numbers", [])

        # Parse teammates into a list of (iracing_name, real_name) tuples
        teammate_map: dict[str, str] = {}  # iracing_name -> real_name
        for entry in explicit_teammates:
            if isinstance(entry, str) and ":" in entry:
                # "Evolution: Patrik Farsang" -> nickname: real_name
                iracing_name, real_name = entry.split(":", 1)
                iracing_name = iracing_name.strip()
                real_name = real_name.strip()
                teammate_map[iracing_name] = real_name
            elif isinstance(entry, str):
                # Plain name — iRacing name = real name
                teammate_map[entry] = entry
            elif isinstance(entry, dict):
                # YAML mapping format: {Evolution: Patrik Farsang}
                for k, v in entry.items():
                    teammate_map[str(k)] = str(v)

        # Store the alias map on the client for later use
        self.driver_aliases = teammate_map

        if teammate_map and self._drivers:
            for d in self._drivers:
                if d.driver_name in teammate_map:
                    team_indices.add(d.car_idx)
                    real_name = teammate_map[d.driver_name]
                    if real_name != d.driver_name:
                        logger.debug(
                            f"Config-matched teammate: {d.driver_name} / {real_name} (#{d.car_number})"
                        )
                    else:
                        logger.debug(
                            f"Config-matched teammate: {d.driver_name} (#{d.car_number})"
                        )

        if explicit_car_numbers and self._drivers:
            for d in self._drivers:
                if d.car_number in explicit_car_numbers:
                    team_indices.add(d.car_idx)
                    logger.debug(
                        f"Config-matched teammate by car number: {d.driver_name} (#{d.car_number})"
                    )

        if team_indices:
            logger.info(f"Detected {len(team_indices)} team cars: {team_indices}")
        else:
            logger.warning(
                "No teammates detected — only player car will have full tracking"
            )

        return team_indices

    # --- Structured accessors ---

    def get_standings(self) -> list[DriverState]:
        """Get current standings for all drivers."""
        if not self.is_connected:
            return []

        standings = []

        # Get arrays of per-car data
        positions = self.get_telemetry_value("CarIdxPosition", [])
        class_positions = self.get_telemetry_value("CarIdxClassPosition", [])
        laps = self.get_telemetry_value("CarIdxLap", [])
        laps_completed = self.get_telemetry_value("CarIdxLapCompleted", [])
        lap_dist_pcts = self.get_telemetry_value("CarIdxLapDistPct", [])
        on_pit_roads = self.get_telemetry_value("CarIdxOnPitRoad", [])
        track_surfaces = self.get_telemetry_value("CarIdxTrackSurface", [])
        best_lap_times = self.get_telemetry_value("CarIdxBestLapTime", [])
        last_lap_times = self.get_telemetry_value("CarIdxLastLapTime", [])
        speeds = self.get_telemetry_value("CarIdxEstTime", [])  # Est time used as proxy
        rpms = self.get_telemetry_value("CarIdxRPM", [])
        gears = self.get_telemetry_value("CarIdxGear", [])
        p2p_statuses = self.get_telemetry_value("CarIdxP2P_Status", [])
        p2p_counts = self.get_telemetry_value("CarIdxP2P_Count", [])
        tire_compounds = self.get_telemetry_value("CarIdxTireCompound", [])
        # Per-car incident counts aren't available as a telemetry array,
        # but we can use CurDriverIncidentCount from session info

        num_cars = len(positions) if isinstance(positions, (list, tuple)) else 0

        for i in range(num_cars):
            try:
                pos = positions[i] if i < len(positions) else 0
                if pos <= 0:
                    continue  # Skip invalid/disconnected cars

                standings.append(
                    DriverState(
                        car_idx=i,
                        position=int(pos) if pos > 0 else 0,
                        class_position=int(class_positions[i])
                        if i < len(class_positions)
                        else 0,
                        lap=int(laps[i]) if i < len(laps) else 0,
                        lap_completed=int(laps_completed[i])
                        if i < len(laps_completed)
                        else 0,
                        lap_dist_pct=float(lap_dist_pcts[i])
                        if i < len(lap_dist_pcts)
                        else 0.0,
                        on_pit_road=bool(on_pit_roads[i])
                        if i < len(on_pit_roads)
                        else False,
                        track_surface=int(track_surfaces[i])
                        if i < len(track_surfaces)
                        else 0,
                        best_lap_time=float(best_lap_times[i])
                        if i < len(best_lap_times)
                        else -1.0,
                        last_lap_time=float(last_lap_times[i])
                        if i < len(last_lap_times)
                        else -1.0,
                        speed=float(speeds[i]) if i < len(speeds) else 0.0,
                        rpm=float(rpms[i]) if i < len(rpms) else 0.0,
                        gear=int(gears[i]) if i < len(gears) else 0,
                        p2p_status=int(p2p_statuses[i]) if i < len(p2p_statuses) else 0,
                        p2p_count=int(p2p_counts[i]) if i < len(p2p_counts) else 0,
                        tire_compound=int(tire_compounds[i])
                        if i < len(tire_compounds)
                        else 0,
                        incidents=self._get_driver_incident_count(i),
                    )
                )
            except (IndexError, TypeError, ValueError):
                continue

        return standings

    def _get_driver_incident_count(self, car_idx: int) -> int:
        """Look up incident count for a car from session info."""
        for d in self._drivers:
            if d.car_idx == car_idx:
                # CurDriverIncidentCount is in the raw session info
                raw = self._session_info_cache or {}
                driver_info = raw.get("DriverInfo", {})
                for driver in driver_info.get("Drivers", []):
                    if int(driver.get("CarIdx", -1)) == car_idx:
                        return int(driver.get("CurDriverIncidentCount", 0))
        return 0

    def get_telemetry_units(self, fields: list[str] | None = None) -> dict[str, str]:
        """Get the unit strings for telemetry variables from iRacing SDK.

        Each VarHeader in iRacing's shared memory has a 'unit' field (e.g.
        'kPa', 'm', 'degC', 'degF', 'L', 'gal', 'bar', 'V', '%' etc).
        This method reads those unit strings so the calling code can convert
        values to the desired display units.

        Args:
            fields: Optional list of field names to query. If None, returns
                units for all known telemetry fields.

        Returns:
            Dict mapping field name -> unit string (e.g. {'LFcoldPressure': 'kPa'}).
        """
        if not self.is_connected:
            return {}

        try:
            var_dict = self._ir._var_headers_dict
        except (AttributeError, TypeError):
            return {}

        if fields is None:
            # All fields that might need unit conversion
            fields = [
                "FuelLevel",
                "FuelLevelPct",
                "FuelUsePerHour",
                "LFcoldPressure",
                "RFcoldPressure",
                "LRcoldPressure",
                "RRcoldPressure",
                "LFbrakeLinePress",
                "RFbrakeLinePress",
                "LRbrakeLinePress",
                "RRbrakeLinePress",
                "OilTemp",
                "OilPress",
                "WaterTemp",
                "ManifoldPress",
                "Voltage",
                "CarDistAhead",
                "CarDistBehind",
                "dcBrakeBias",
                "LFtempCL",
                "LFtempCM",
                "LFtempCR",
                "RFtempCL",
                "RFtempCM",
                "RFtempCR",
                "LRtempCL",
                "LRCtempCM",
                "LRtempCR",
                "RRtempCL",
                "RRtempCM",
                "RRtempCR",
                "LFodometer",
                "RFodometer",
                "LRodometer",
                "RRodometer",
                "TrackTemp",
                "AirTemp",
                "AirPressure",
                "WindVel",
                "Speed",
            ]

        units: dict[str, str] = {}
        for field in fields:
            try:
                var_header = var_dict.get(field)
                if var_header and hasattr(var_header, "unit"):
                    unit_str = var_header.unit
                    if unit_str:
                        units[field] = unit_str.strip("\x00").strip()
            except Exception:
                pass

        return units

    def get_player_telemetry(self) -> dict[str, Any]:
        """Get telemetry values specific to the player's car.

        Returns a dict with player-specific fields like fuel, tyre temps,
        brake data, damage, etc.
        """
        if not self.is_connected:
            return {}

        fields = [
            # Car state
            "Speed",
            "Gear",
            "RPM",
            "Throttle",
            "Brake",
            "Clutch",
            "SteeringWheelAngle",
            "SteeringWheelTorque",
            # Fuel
            "FuelLevel",
            "FuelLevelPct",
            "FuelUsePerHour",
            # Engine health
            "OilPress",
            "OilTemp",
            "OilLevel",
            "WaterTemp",
            "WaterLevel",
            "FuelPress",
            "EngineWarnings",
            "ManifoldPress",
            "Voltage",
            # Player position
            "PlayerCarIdx",
            "PlayerCarPosition",
            "PlayerCarClassPosition",
            "CarDistAhead",
            "CarDistBehind",
            "CarLeftRight",
            "PlayerCarMyIncidentCount",
            "PlayerCarTeamIncidentCount",
            "PlayerCarTowTime",
            # Car status
            "IsOnTrack",
            "IsInGarage",
            "EnterExitReset",
            # Laps
            "Lap",
            "LapCompleted",
            "LapCurrentLapTime",
            "LapDistPct",
            "LapBestLapTime",
            "LapBestLap",
            "LapLastLapTime",
            "LapDeltaToBestLap",
            "LapDeltaToBestLap_OK",
            "LapDeltaToSessionBestLap",
            "LapDeltaToOptimalLap",
            # Tyre temps (per corner, per zone)
            "LFtempCL",
            "LFtempCM",
            "LFtempCR",
            "RFtempCL",
            "RFtempCM",
            "RFtempCR",
            "LRtempCL",
            "LRtempCM",
            "LRtempCR",
            "RRtempCL",
            "RRtempCM",
            "RRtempCR",
            # Tyre pressures
            "LFcoldPressure",
            "RFcoldPressure",
            "LRcoldPressure",
            "RRcoldPressure",
            # Tyre wear (per corner, per zone)
            "LFwearL",
            "LFwearM",
            "LFwearR",
            "RFwearL",
            "RFwearM",
            "RFwearR",
            "LRwearL",
            "LRwearM",
            "LRwearR",
            "RRwearL",
            "RRwearM",
            "RRwearR",
            # Tyre odometer (per corner)
            "LFodometer",
            "RFodometer",
            "LRodometer",
            "RRodometer",
            # Tyre compound and sets
            "PlayerTireCompound",
            "PlayerCarDryTireSetLimit",
            "TireSetsAvailable",
            "TireSetsUsed",
            "FrontTireSetsAvailable",
            "FrontTireSetsUsed",
            "LeftTireSetsAvailable",
            "LeftTireSetsUsed",
            "RightTireSetsAvailable",
            "RightTireSetsUsed",
            "RearTireSetsAvailable",
            "RearTireSetsUsed",
            # Brake temps
            "LFbrakeLinePress",
            "RFbrakeLinePress",
            "LRbrakeLinePress",
            "RRbrakeLinePress",
            "Brake",
            "BrakeABSactive",
            "dcBrakeBias",
            # Damage and penalties
            "PlayerCarWeightPenalty",
            "PlayerFastRepairsUsed",
            "PitRepairLeft",
            "PitOptRepairLeft",
            "PlayerIncidents",
            # Pit
            "OnPitRoad",
            "PitstopActive",
            "PitsOpen",
            "PlayerCarPitSvStatus",
            "PlayerCarInPitStall",
            "PitSvFlags",
            "PitSvFuel",
            "PitSvLFP",
            "PitSvRFP",
            "PitSvLRP",
            "PitSvRRP",
            "PitSvTireCompound",
            "FastRepairAvailable",
            "FastRepairUsed",
            "dpFuelFill",
            "dpFuelAddKg",
            "dpFuelAutoFillActive",
            "dpFuelAutoFillEnabled",
            "dpFastRepair",
            "dpTireChange",
            "dpLFTireChange",
            "dpRFTireChange",
            "dpLRTireChange",
            "dpRRTireChange",
            "dpLFTireColdPress",
            "dpRFTireColdPress",
            "dpLRTireColdPress",
            "dpRRTireColdPress",
            # Session
            "SessionFlags",
            "CarIdxSessionFlags",
            "SessionLapsRemain",
            "SessionLapsRemainEx",
            "SessionTimeRemain",
            "SessionTime",
            "SessionNum",
            "SessionState",
            "RaceLaps",
            # Push to pass
            "P2P_Status",
            "P2P_Count",
            "PushToPass",
            # Shift lights
            "ShiftIndicatorPct",
            "PlayerCarSLShiftRPM",
            "PlayerCarSLFirstRPM",
            "PlayerCarSLLastRPM",
            "PlayerCarSLBlinkRPM",
            # G-forces
            "LatAccel",
            "LongAccel",
            "VertAccel",
            # Track conditions
            "TrackTemp",
            "TrackTempCrew",
            "TrackWetness",
            "WeatherDeclaredWet",
            "PlayerTrackSurface",
            "PlayerTrackSurfaceMaterial",
            "AirTemp",
            "AirPressure",
            "AirDensity",
            "Precipitation",
            "WindDir",
            "WindVel",
            "Skies",
            "FogLevel",
            "RelativeHumidity",
        ]

        result: dict[str, Any] = {}
        for field in fields:
            val = self.get_telemetry_value(field)
            if val is not None:
                result[field] = val
        return result

    def get_session_summary(self) -> SessionInfo:
        """Get a structured session info object."""
        info = self.get_session_info()
        if not info:
            return SessionInfo()

        weekend = info.get("WeekendInfo", {})
        session_info = info.get("SessionInfo", {})
        sessions = session_info.get("Sessions", [])
        current_session_num = self.get_telemetry_value("SessionNum", 0)

        # Find the current session from the list
        current_session = {}
        for s in sessions:
            if s.get("SessionNum") == current_session_num:
                current_session = s
                break

        # Get session details
        session_type = current_session.get("SessionType", "")
        session_name = current_session.get("SessionName", "")

        # Get results/laps info from telemetry (more reliable)
        laps_total = self.get_telemetry_value("SessionLapsRemainEx", -1)
        if laps_total == -1:
            laps_total = self.get_telemetry_value("SessionLapsRemain", 0)

        return SessionInfo(
            track_name=weekend.get("TrackName", ""),
            track_config=weekend.get("TrackConfigName", ""),
            session_type=session_type,
            session_num=current_session_num,
            session_name=session_name,
            laps_total=self.get_telemetry_value("RaceLaps", 0),
            laps_remain=self.get_telemetry_value("SessionLapsRemain", 0),
            time_remain=self.get_telemetry_value("SessionTimeRemain", 0.0),
            flags=int(self.get_telemetry_value("SessionFlags", 0)),
            player_flags=int(
                self.get_telemetry_value("CarIdxSessionFlags", [0])[
                    self.get_telemetry_value("PlayerCarIdx", 0)
                ]
            ),
            session_state=int(self.get_telemetry_value("SessionState", 0)),
        )

    @property
    def drivers(self) -> list[DriverInfo]:
        """Cached list of driver info from session info."""
        return self._drivers

    def get_driver_name(self, car_idx: int) -> str:
        """Look up driver name by car index."""
        for d in self._drivers:
            if d.car_idx == car_idx:
                return d.driver_name
        return f"Car #{car_idx}"

    # --- Command methods (for action_executor) ---

    def pit_command(self, command: str, param: int = 0):
        """Send a pit command to iRacing.

        Args:
            command: Pit command mode name (e.g. 'fuel', 'lf', 'clear_tires').
            param: Optional parameter (e.g. fuel amount in litres).

        Available commands from PitCommandMode:
            clear, clear_fr, clear_fuel, clear_tires, clear_ws,
            fr (fast repair), fuel, lf, lr, rf, rr, ws (windshield)
        """
        try:
            mode = getattr(irsdk.PitCommandMode, command, None)
            if mode is None:
                logger.error(f"Unknown pit command: {command}")
                return
            self._ir.pit_command(pit_command_mode=mode, var=param)
            logger.info(f"Sent pit command: {command} param={param}")
        except Exception as e:
            logger.error(f"Failed to send pit command {command}: {e}")

    def chat_command(self, command_name: str = "begin_chat"):
        """Send a chat command to iRacing.

        Args:
            command_name: Chat command mode name.
                begin_chat, cancel, macro, reply
        """
        try:
            mode = getattr(irsdk.ChatCommandMode, command_name, None)
            if mode is None:
                logger.error(f"Unknown chat command: {command_name}")
                return
            self._ir.chat_command(chat_command_mode=mode)
            logger.info(f"Sent chat command: {command_name}")
        except Exception as e:
            logger.error(f"Failed to send chat command {command_name}: {e}")

    def broadcast_pit_command(self, command_name: str, param: int = 0):
        """Send a broadcast pit command via pyirsdk's internal method.

        This is an alternative to pit_command that uses the broadcast
        message approach for commands not covered by PitCommandMode.
        """
        try:
            mode = getattr(irsdk.PitCommandMode, command_name, None)
            if mode is not None:
                self._ir.pit_command(pit_command_mode=mode, var=param)
            else:
                logger.warning(f"Unknown broadcast command: {command_name}")
        except Exception as e:
            logger.error(f"Failed to broadcast command {command_name}: {e}")
