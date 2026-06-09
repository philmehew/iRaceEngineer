"""
Context builder — condenses race state into a human-readable prompt
for the LLM. Configurable depth: minimal, medium, full.
"""

import logging

logger = logging.getLogger(__name__)


def format_lap_time(seconds: float) -> str:
    """Format a lap time in seconds to M:SS.mmm format."""
    if seconds <= 0:
        return "N/A"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


def format_gap(seconds: float) -> str:
    """Format a time gap in seconds to a human-readable string."""
    if seconds == 0:
        return "0.000s"
    sign = "+" if seconds > 0 else "-"
    return f"{sign}{abs(seconds):.3f}s"


def format_temp(temp: float) -> str:
    """Format a temperature value to 1 decimal place."""
    if temp <= 0:
        return "N/A"
    return f"{temp:.1f}°C"


def format_pct(value: float) -> str:
    """Format a percentage (0-1) as a whole number percent."""
    if value <= 0:
        return "N/A"
    return f"{value * 100:.0f}%"


def format_wear(wear: float) -> str:
    """Format tyre life remaining (0-1) as a percentage.

    iRacing reports wear as 1.0=new, 0.0=worn out, so higher = more life left.
    The label 'life' makes this unambiguous to the LLM.
    """
    if wear <= 0:
        return "N/A"
    return f"life {wear * 100:.1f}%"


def format_metric(value: float, unit: str = "", decimals: int = 2) -> str:
    """Format a numeric metric to a consistent number of decimal places."""
    if value == 0:
        return "N/A"
    return f"{value:.{decimals}f}{unit}"


def format_engine_warnings(warnings: int) -> str:
    """Format EngineWarnings bitmask to human-readable labels."""
    if warnings == 0:
        return ""
    labels = []
    # iRacing EngineWarnings bits (from SDK docs)
    if warnings & 0x01:
        labels.append("water temp")
    if warnings & 0x02:
        labels.append("fuel pressure")
    if warnings & 0x04:
        labels.append("oil pressure")
    if warnings & 0x08:
        labels.append("engine stall")
    if warnings & 0x10:
        labels.append("pit speed limiter")
    if warnings & 0x20:
        labels.append("rev limiter")
    if warnings & 0x40:
        labels.append("fuel level")
    if warnings & 0x80:
        labels.append("oil temp")
    # Bits 8+ are less common
    if warnings & ~0xFF:
        labels.append(f"other(0x{warnings & ~0xFF:x})")
    return ", ".join(labels)


def format_car_proximity(car_left_right: int) -> str:
    """Format CarLeftRight value to human-readable string."""
    if car_left_right == 0:
        return ""
    parts = []
    if car_left_right & 1:
        parts.append("car LEFT")
    if car_left_right & 2:
        parts.append("car RIGHT")
    return " | ".join(parts)


# iRacing TrackWetness scale (integer, not percentage)
TRACK_WETNESS_LABELS = {0: "Dry", 1: "Damp", 2: "Wet", 3: "Very Wet"}


class ContextBuilder:
    """Builds LLM-ready prompts from race state snapshots.

    Supports three context depths:
    - minimal: position, fuel, laps remaining (~200 bytes)
    - medium: + gaps, tyre temps, flags (~500 bytes)
    - full:   + lap trends, nearby cars, pit window, weather, damage (~1-2KB)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        prompt_config = self.config.get("prompt", {})
        self.context_depth = prompt_config.get("context_depth", "full")
        self.system_prompt = prompt_config.get("system", self._default_system_prompt())
        self.include_lap_history = prompt_config.get("include_lap_history", 5)
        self.include_nearby_cars = prompt_config.get("include_nearby_cars", 3)

    def _default_system_prompt(self) -> str:
        return (
            "You are a race engineer talking to your driver over radio. "
            "Speak in short, clear sentences like a real race engineer — "
            "the driver is in a race and needs fast, spoken-style answers. "
            "No markdown, no bullet points, no bold, no headers. "
            "Just plain sentences, like you're on the radio. "
            "Example: 'Box this lap. Tyres are gone. Add 60 litres.' "
            "If you're unsure, say so. Never invent data not provided. "
            "The 'Your car' section is the driver you are talking to.\n\n"
            "You may optionally include action directives in your response using the "
            "format [ACTION] action_name[: parameter]. These will be executed (when "
            "enabled) or logged (in dry-run mode). Available actions:\n"
            "- [ACTION] pit_this_lap\n"
            "- [ACTION] add_fuel: <litres>\n"
            "- [ACTION] change_tyres\n"
            "- [ACTION] clear_penalty"
        )

    def build_prompt(self, state: dict, question: str = "") -> list[dict]:
        """Build the messages list for the OpenAI chat completions API.

        Args:
            state: Race state snapshot from RaceState.get_snapshot()
            question: Optional driver question to append

        Returns:
            List of message dicts for the OpenAI API.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        if self.context_depth == "minimal":
            user_content = self._build_minimal(state)
        elif self.context_depth == "medium":
            user_content = self._build_medium(state)
        else:  # full
            user_content = self._build_full(state)

        if question:
            user_content += f"\n\nQuestion: {question}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_minimal(self, state: dict) -> str:
        """Build minimal context: position, fuel, laps remaining."""
        lines = []

        session = state.get("session", {})
        player = state.get("player", {})

        # Driver identity
        driver_name = player.get("driver_name", "Driver")

        # Session line
        track = session.get("track_name", "Unknown")
        laps_remain = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        pos = player.get("position", "?")

        lines.append(
            f"Driver: {driver_name}. Race: {track}, P{pos}, Lap {race_laps}/{race_laps + laps_remain if isinstance(laps_remain, int) and isinstance(race_laps, int) else '?'}"
        )

        # Fuel
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        lines.append(f"Fuel: {format_pct(fuel_pct)} (~{fuel_laps:.2f} laps remaining)")
        lines.append(f"Laps remaining: {laps_remain}")

        return "\n".join(lines)

    def _build_medium(self, state: dict) -> str:
        """Build medium context: minimal + gaps, tyre temps, flags, engine health."""
        lines = []
        session = state.get("session", {})
        player = state.get("player", {})

        # Driver identity
        driver_name = player.get("driver_name", "Driver")

        # Session header
        track = session.get("track_name", "Unknown")
        config = session.get("track_config", "")
        track_str = f"{track} {config}".strip() if config else track
        laps_remain = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        total_laps = (
            race_laps + laps_remain
            if isinstance(race_laps, int) and isinstance(laps_remain, int)
            else "?"
        )
        pos = player.get("position", "?")
        class_pos = player.get("class_position", "?")
        flags = session.get("flags", ["Green"])

        lines.append(
            f"Driver: {driver_name}. Race: {track_str}, Lap {race_laps}/{total_laps}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Weather
        weather = session.get("weather", {})
        if weather:
            track_temp = weather.get("track_temp", 0)
            air_temp = weather.get("air_temp", 0)
            wetness = weather.get("track_wetness", 0)
            wet_str = ""
            if wetness > 0:
                wet_label = TRACK_WETNESS_LABELS.get(
                    int(wetness), f"level {int(wetness)}"
                )
                wet_str = f", {wet_label}"
            if track_temp > 0 or air_temp > 0:
                lines.append(
                    f"Track: {format_temp(track_temp)}, Air: {format_temp(air_temp)}{wet_str}"
                )

        # Player car
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_level = player.get("fuel_level", 0)
        lines.append("\nYour car:")
        lines.append(
            f"  Fuel: {fuel_level:.2f}L ({format_pct(fuel_pct)}), ~{fuel_laps:.2f} laps remaining"
        )

        # Engine warnings (critical for race engineering)
        engine_warnings = player.get("engine_warnings", 0)
        if engine_warnings:
            warning_str = format_engine_warnings(engine_warnings)
            lines.append(f"  ⚠ ENGINE WARNING: {warning_str}")

        # Tyre temps
        tyres = player.get("tyres", {})
        if tyres:
            tyre_strs = []
            for corner in ["LF", "RF", "LR", "RR"]:
                ts = tyres.get(corner, {})
                avg_temp = ts.get("temp_center", 0)
                if avg_temp > 0:
                    tyre_strs.append(f"{corner} {format_temp(avg_temp)}")
            if tyre_strs:
                lines.append(f"  Tyres: {' / '.join(tyre_strs)}")

        # Incidents
        incidents = player.get("incidents", 0)
        if incidents > 0:
            lines.append(f"  Incidents: {incidents}x")

        return "\n".join(lines)

    def _build_full(self, state: dict) -> str:
        """Build full context: everything a race engineer needs."""
        lines = []
        session = state.get("session", {})
        player = state.get("player", {})
        config = session.get("config", {})

        # === Session header ===
        track = session.get("track_name", "Unknown")
        config_name = session.get("track_config", "")
        track_str = f"{track} {config_name}".strip() if config_name else track
        laps_remain = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        total_laps = (
            race_laps + laps_remain
            if isinstance(race_laps, int) and isinstance(laps_remain, int)
            else "?"
        )
        pos = player.get("position", "?")
        class_pos = player.get("class_position", "?")
        flags = session.get("flags", ["Green"])
        driver_name = player.get("driver_name", "Driver")

        lines.append(
            f"Driver: {driver_name}. Race: {track_str}, Lap {race_laps}/{total_laps}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Track config
        track_length = config.get("track_length_km", 0)
        track_turns = config.get("track_num_turns", 0)
        pit_speed = config.get("pit_speed_limit_kph", 0)
        if track_length or track_turns:
            track_info = []
            if track_length:
                track_info.append(f"{track_length:.1f}km")
            if track_turns:
                track_info.append(f"{track_turns} turns")
            if pit_speed:
                track_info.append(f"pit limit {pit_speed:.1f}kph")
            lines.append(f"Track: {' | '.join(track_info)}")

        # Session config
        is_fixed = config.get("is_fixed_setup", False)
        incident_limit = config.get("incident_limit", "")
        fast_repairs = config.get("fast_repairs_limit", "")
        fuel_max = config.get("fuel_max_litres", 0)
        if is_fixed or incident_limit or fuel_max:
            cfg_parts = []
            if is_fixed:
                cfg_parts.append("FIXED SETUP")
            if incident_limit:
                cfg_parts.append(f"incidents: {incident_limit}")
            if fast_repairs:
                cfg_parts.append(f"fast repairs: {fast_repairs}")
            if fuel_max:
                cfg_parts.append(f"tank: {fuel_max:.0f}L")
            lines.append(f"Session: {' | '.join(cfg_parts)}")

        # Weather
        weather = session.get("weather", {})
        if weather:
            track_temp = weather.get("track_temp", 0)
            air_temp = weather.get("air_temp", 0)
            wetness = weather.get("track_wetness", 0)
            declared_wet = weather.get("weather_declared_wet", False)
            precipitation = weather.get("precipitation", 0)
            wind_vel = weather.get("wind_vel", 0)

            weather_parts = []
            if track_temp > 0:
                weather_parts.append(f"Track {format_temp(track_temp)}")
            if air_temp > 0:
                weather_parts.append(f"Air {format_temp(air_temp)}")
            if wetness > 0:
                wet_label = TRACK_WETNESS_LABELS.get(
                    int(wetness), f"level {int(wetness)}"
                )
                weather_parts.append(f"track {wet_label}")
            if declared_wet:
                weather_parts.append("DECLARED WET")
            if precipitation > 0:
                weather_parts.append(f"rain {precipitation:.0%}")
            if wind_vel > 0.5:
                weather_parts.append(f"wind {wind_vel:.2f}m/s")
            if weather_parts:
                lines.append(f"Weather: {' | '.join(weather_parts)}")

        # === Engine health ===
        oil_temp = player.get("oil_temp", 0)
        oil_press = player.get("oil_press", 0)
        water_temp = player.get("water_temp", 0)
        voltage = player.get("voltage", 0)
        engine_warnings = player.get("engine_warnings", 0)
        manifold_press = player.get("manifold_press", 0)

        engine_parts = []
        if oil_temp > 0:
            engine_parts.append(f"Oil {format_temp(oil_temp)}")
        if oil_press > 0:
            engine_parts.append(f"OilP {oil_press:.2f}bar")
        if water_temp > 0:
            engine_parts.append(f"Water {format_temp(water_temp)}")
        if voltage > 0:
            engine_parts.append(f"{voltage:.2f}V")
        if manifold_press > 0 and manifold_press < 5:  # Only show if not default-ish
            engine_parts.append(f"Manifold {manifold_press:.2f}bar")
        if engine_parts:
            lines.append(f"Engine: {' | '.join(engine_parts)}")

        if engine_warnings:
            warning_str = format_engine_warnings(engine_warnings)
            lines.append(f"  ⚠ ENGINE WARNING: {warning_str}")

        # === Player car ===
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_level = player.get("fuel_level", 0)
        fuel_rate = player.get("fuel_use_per_hour", 0)

        lines.append("\nYour car:")
        on_track = player.get("is_on_track", True)
        in_garage = player.get("is_in_garage", False)
        status_parts = []
        if not on_track:
            status_parts.append("OFF TRACK")
        if in_garage:
            status_parts.append("IN GARAGE")
        if status_parts:
            lines.append(f"  Status: {' | '.join(status_parts)}")

        lines.append(f"  Position: P{pos} (Class P{class_pos})")
        # Fuel display — avoid showing "0 laps remaining" when we have fuel
        # (happens early race before lap history is established)
        if fuel_laps > 0:
            lines.append(
                f"  Fuel: {fuel_level:.2f}L ({format_pct(fuel_pct)}), ~{fuel_laps:.1f} laps remaining"
            )
        else:
            lines.append(f"  Fuel: {fuel_level:.2f}L ({format_pct(fuel_pct)})")
        if fuel_max and fuel_level > 0:
            lines.append(
                f"  Tank capacity: {fuel_max:.1f}L ({fuel_level / fuel_max * 100:.0f}% full)"
            )
        if fuel_rate > 0:
            lines.append(f"  Fuel burn rate: {fuel_rate:.2f} L/hr")

        # Tyres
        tyres = player.get("tyres", {})
        tyre_odometers = player.get("tyre_odometers", {})
        if tyres:
            lines.append("  Tyres:")
            for corner in ["LF", "RF", "LR", "RR"]:
                ts = tyres.get(corner, {})
                if not ts:
                    continue
                avg_temp = ts.get("temp_center", 0)
                pressure = ts.get("cold_pressure", 0)
                wear_center = ts.get("wear_center", 0)
                odo = tyre_odometers.get(corner, 0)
                parts = [f"{format_temp(avg_temp)}"]
                if pressure > 0:
                    parts.append(f"{pressure:.2f}PSI")
                if wear_center > 0:
                    parts.append(f"{format_wear(wear_center)}")
                if odo > 0:
                    parts.append(f"{odo:.0f}km")
                lines.append(f"    {corner}: {' | '.join(parts)}")

        # Brakes + brake bias
        brake_pressures = player.get("brake_pressures", {})
        brake_temps = []
        for corner in ["LF", "RF", "LR", "RR"]:
            bp = brake_pressures.get(corner, 0)
            if bp > 0:
                brake_temps.append(f"{corner} {bp:.0f}")
        if brake_temps:
            lines.append(f"  Brakes: {' / '.join(brake_temps)}")
        brake_bias = player.get("brake_bias", 0)
        if brake_bias:
            lines.append(f"  Brake bias: {brake_bias:.2f}%")

        # Damage / incidents / penalties
        incidents = player.get("incidents", 0)
        team_incidents = player.get("team_incidents", 0)
        weight_penalty = player.get("weight_penalty", 0)
        fast_repairs_used = player.get("fast_repairs_used", 0)
        repair_time = player.get("pit_repair_time_left", 0)
        opt_repair_time = player.get("pit_opt_repair_time_left", 0)

        damage_parts = []
        if incidents > 0 or team_incidents > 0:
            damage_parts.append(f"incidents {incidents}x (team {team_incidents}x)")
        if weight_penalty > 0:
            damage_parts.append(f"+{weight_penalty:.2f}kg damage weight")
        if fast_repairs_used > 0:
            damage_parts.append(f"{fast_repairs_used} fast repairs used")
        if repair_time > 0:
            damage_parts.append(f"{repair_time:.0f}s repair time needed")
        if opt_repair_time > 0:
            damage_parts.append(f"{opt_repair_time:.0f}s opt repair time")
        if damage_parts:
            lines.append(f"  Damage: {' | '.join(damage_parts)}")

        # Push-to-pass
        p2p_remaining = player.get("p2p_remaining", 0)
        p2p_active = player.get("p2p_active", False)
        if p2p_remaining > 0:
            lines.append(
                f"  Push-to-pass: {p2p_remaining} remaining{' (ACTIVE)' if p2p_active else ''}"
            )

        # Shift lights
        shift_pct = player.get("shift_indicator_pct", 0)
        shift_rpm = player.get("shift_rpm", 0)
        if shift_rpm > 0 or shift_pct > 0:
            shift_str = []
            if shift_rpm > 0:
                shift_str.append(f"shift at {shift_rpm:.0f}rpm")
            if shift_pct > 0:
                shift_str.append(f"{shift_pct:.0%} throttle")
            lines.append(f"  Shift: {' | '.join(shift_str)}")

        # Proximity
        car_prox = player.get("car_left_right", 0)
        dist_ahead = player.get("car_dist_ahead", 0)
        dist_behind = player.get("car_dist_behind", 0)
        tow_time = player.get("tow_time", 0)
        prox_parts = []
        if car_prox:
            prox_str = format_car_proximity(car_prox)
            if prox_str:
                prox_parts.append(prox_str)
        if dist_ahead > 0:
            prox_parts.append(f"+{dist_ahead:.2f}s ahead")
        if dist_behind > 0:
            prox_parts.append(f"-{dist_behind:.2f}s behind")
        if tow_time > 0:
            prox_parts.append(f"tow {tow_time:.2f}s")
        if prox_parts:
            lines.append(f"  Proximity: {' | '.join(prox_parts)}")

        # Lap history
        lap_history = state.get("lap_history", [])
        if lap_history:
            recent = lap_history[-self.include_lap_history :]
            times = [format_lap_time(r["lap_time"]) for r in recent]
            fuels = [
                f"{r['fuel_used']:.1f}L" for r in recent if r.get("fuel_used", 0) > 0
            ]
            trend = player.get("lap_time_trend", 0)

            lap_str = ", ".join(times)
            if fuels:
                lap_str += f" (fuel: {', '.join(fuels)})"

            lines.append(f"  Last {len(recent)} laps: {lap_str}")
            if abs(trend) > 0.01:
                direction = "slowing" if trend > 0 else "gaining"
                lines.append(f"  Trend: {direction} {abs(trend):.2f}s/lap")

        # === Pit status ===
        pit_lines = []
        if player.get("pits_open"):
            pit_lines.append("Pits open")
        if player.get("on_pit_road"):
            pit_lines.append("ON PIT ROAD")
        if player.get("pitstop_active"):
            pit_lines.append("Pit stop in progress")
        if player.get("fast_repair_available"):
            pit_lines.append("Fast repair available")
        if player.get("tire_sets_available", 0) > 0:
            pit_lines.append(f"Tire sets available: {player['tire_sets_available']}")
        if pit_lines:
            lines.append(f"\nPit: {' | '.join(pit_lines)}")

        # === Nearby cars ===
        nearby = state.get("nearby_cars", [])
        if nearby:
            lines.append("\nNearby:")
            for car in nearby[: self.include_nearby_cars]:
                name = car.get("driver_name", f"P{car.get('position', '?')}")
                pos = car.get("position", "?")
                gap = car.get("gap_seconds", 0)
                lap_time = car.get("last_lap_time", -1)
                p2p = "P2P available" if car.get("p2p_available") else ""
                on_pit = " (in pits)" if car.get("on_pit_road") else ""

                car_str = f"  P{pos} ({name}): {format_gap(gap)}"
                if lap_time > 0:
                    car_str += f", last lap {format_lap_time(lap_time)}"
                if p2p:
                    car_str += f", {p2p}"
                if on_pit:
                    car_str += on_pit
                lines.append(car_str)

        return "\n".join(lines)

    def build_action_prompt_addition(
        self, available_actions: list[str] | None = None
    ) -> str:
        """Return additional prompt text describing available actions.

        Args:
            available_actions: List of action names the LLM can use.
                If None, uses the default set.
        """
        if available_actions is None:
            available_actions = [
                "pit_this_lap",
                "add_fuel: <litres>",
                "change_tyres",
                "clear_penalty",
            ]

        actions_str = "\n".join(f"  - [ACTION] {a}" for a in available_actions)
        return (
            f"\n\nYou may include action directives in your response using this format:\n"
            f"{actions_str}\n\n"
            f"Actions are optional — give text advice unless an action is clearly warranted."
        )
