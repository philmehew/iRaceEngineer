"""
Context builder — condenses race state into a human-readable prompt
for the LLM. Configurable depth: minimal, medium, full.
"""

import logging

logger = logging.getLogger(__name__)

# Corner abbreviation → display name (data keys stay as abbreviations)
CORNER_NAMES = {
    "LF": "Left Front",
    "RF": "Right Front",
    "LR": "Left Rear",
    "RR": "Right Rear",
}


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


def kpa_to_psi(kpa: float) -> float:
    """Convert kPa to PSI."""
    return kpa * 0.145038


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
    """Format CarLeftRight value to human-readable string.

    iRacing CarLeftRight is an ordinal enum (not bitmask):
        0=off, 1=clear, 2=car left, 3=car right, 4=both, 5=two left, 6=two right
    """
    if car_left_right in (0, 1):
        return ""
    parts = []
    if car_left_right in (2, 4, 5):
        parts.append("car LEFT")
    if car_left_right in (3, 4, 6):
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
            "You are a race engineer. Speak in short radio-style sentences. No markdown, bullets, or headers.\n"
            "Example: 'Box this lap. Tyres are gone. Add 60 litres.'\n"
            "If unsure, say so. Never invent data. The 'Your car' section is the driver you're talking to.\n\n"
            "Rules:\n"
            "- Only respond when asked. Don't promise to monitor or call pit later.\n"
            "- Don't suggest setup changes (pressures, brake bias) without known reference ranges.\n"
            "- Don't assess temps or pressures as high/low/normal/stable without knowing the car's normal range — just report values.\n"
            "- Don't name track corners unless that data is in the context.\n"
            "- 'Incidents' are safety points (0x per off-track), NOT car damage. Always report the count when it's in the context.\n"
            "- If tyre data is marked unreliable, still report the values but don't comment on degradation, trends, or whether they're changing. Say 'last known: Left Front 79C...' rather than skipping them.\n"
            "- [ACTION] add_fuel amounts: whole litres only (integers, never decimals), min 1L, max = tank capacity minus current fuel. Never exceed tank capacity.\n"
            "- Weather data is current conditions only — never predict future weather.\n"
            "- Lap times in the 'Pace' line are the driver's own times. Lap times in the 'Nearby' section are other cars' times.\n"
            "- When fuel burn says 'unknown', do NOT invent a specific L/lap figure. Say 'fuel burn unknown' or estimate from race structure only.\n"
            "- Never say 'monitor' or 'keep an eye on' — you give one-shot advice, not continuous tracking.\n"
            "- Only include [ACTION] when the driver asks about pitting, fuel, tyres, or strategy. Do not add actions to unrelated questions.\n"
            "- When 'Fuel to add' is shown in context, use that exact amount in [ACTION] add_fuel. Do not invent different amounts.\n\n"
            "Optional actions: [ACTION] pit_this_lap | add_fuel: <litres> | change_tyres | clear_penalty"
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
        estimated_total = state.get("estimated_total_laps")
        LAPS_REMAIN_SENTINEL = 32767
        is_time_race = (
            isinstance(laps_remain, int) and laps_remain >= LAPS_REMAIN_SENTINEL
        )

        if is_time_race and estimated_total:
            total_laps_str = f"~{estimated_total}"
        elif (
            isinstance(race_laps, int)
            and isinstance(laps_remain, int)
            and laps_remain < LAPS_REMAIN_SENTINEL
        ):
            total_laps_str = str(race_laps + laps_remain)
        else:
            total_laps_str = "?"

        pos = player.get("position", "?")

        lines.append(
            f"Driver: {driver_name}. Race: {track}, P{pos}, Lap {race_laps}/{total_laps_str}"
        )

        # Fuel with quality indicator
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_est_quality = player.get("fuel_est_quality", "unreliable")
        if fuel_laps > 0 and fuel_est_quality == "good":
            lines.append(f"Fuel: {format_pct(fuel_pct)} (~{fuel_laps:.1f} laps)")
        elif fuel_laps > 0 and fuel_est_quality == "rough":
            lines.append(
                f"Fuel: {format_pct(fuel_pct)} (~{fuel_laps:.1f} laps, approx)"
            )
        elif fuel_pct > 0:
            lines.append(f"Fuel: {format_pct(fuel_pct)} (laps remaining: unreliable)")
        else:
            lines.append(f"Fuel: {format_pct(fuel_pct)}")
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
        track_config = session.get("track_config", "")
        track_str = f"{track} {track_config}".strip() if track_config else track
        session_config = session.get("config", {})
        laps_remain = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        estimated_total = state.get("estimated_total_laps")
        LAPS_REMAIN_SENTINEL = 32767
        is_time_race = (
            isinstance(laps_remain, int) and laps_remain >= LAPS_REMAIN_SENTINEL
        )

        if is_time_race and estimated_total:
            total_laps_str = f"~{estimated_total}"
        elif (
            isinstance(race_laps, int)
            and isinstance(laps_remain, int)
            and laps_remain < LAPS_REMAIN_SENTINEL
        ):
            total_laps_str = str(race_laps + laps_remain)
        else:
            total_laps_str = "?"

        pos = player.get("position", "?")
        class_pos = player.get("class_position", "?")
        flags = session.get("flags", ["Green"])

        lines.append(
            f"Driver: {driver_name}. Race: {track_str}, Lap {race_laps}/{total_laps_str}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Race fastest lap
        race_fastest = player.get("race_fastest_lap")
        if race_fastest and race_fastest.get("time", 0) > 0:
            fast_time = format_lap_time(race_fastest["time"])
            fast_driver = race_fastest.get("driver_name", "?")
            lines.append(f"Race fastest: {fast_time} ({fast_driver})")

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
        fuel_est_quality = player.get("fuel_est_quality", "unreliable")
        fuel_max = session_config.get("fuel_max_litres", 0)
        fuel_max_pct = session_config.get("fuel_max_pct", 1.0)
        fuel_max_start = session_config.get("fuel_max_start_litres", 0)
        effective_fuel_max = fuel_max_start if fuel_max_start > 0 else fuel_max
        lines.append("\nYour car:")
        # Build fuel line with max add if tank capacity is known
        if effective_fuel_max and fuel_level > 0:
            max_add = effective_fuel_max - fuel_level
            if fuel_max_pct < 1.0 and fuel_max_start > 0:
                fuel_str = f"  Fuel: {fuel_level:.2f}L/{fuel_max_start:.1f}L (max add: {max_add:.0f}L, {format_pct(fuel_pct)}"
            else:
                fuel_str = f"  Fuel: {fuel_level:.2f}L/{fuel_max:.0f}L (max add: {max_add:.0f}L, {format_pct(fuel_pct)}"
        else:
            fuel_str = f"  Fuel: {fuel_level:.2f}L ({format_pct(fuel_pct)}"
        if fuel_laps > 0 and fuel_est_quality == "good":
            lines.append(f"{fuel_str}), ~{fuel_laps:.1f} laps")
        elif fuel_laps > 0 and fuel_est_quality == "rough":
            lines.append(f"{fuel_str}), ~{fuel_laps:.1f} laps (approx)")
        elif fuel_pct > 0:
            lines.append(f"{fuel_str}), laps remaining: unreliable")
        else:
            lines.append(f"{fuel_str})")

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
                    tyre_strs.append(f"{CORNER_NAMES[corner]} {format_temp(avg_temp)}")
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
        laps_remain_raw = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        estimated_total = state.get("estimated_total_laps")
        LAPS_REMAIN_SENTINEL = 32767  # iRacing sentinel for unlimited/time-based
        is_time_race = (
            isinstance(laps_remain_raw, int) and laps_remain_raw >= LAPS_REMAIN_SENTINEL
        )

        # Compute laps remaining for fuel calculations
        if is_time_race and estimated_total and isinstance(race_laps, int):
            race_laps_remain = max(0, estimated_total - race_laps)
        elif (
            not is_time_race
            and isinstance(laps_remain_raw, int)
            and laps_remain_raw < LAPS_REMAIN_SENTINEL
        ):
            race_laps_remain = max(0, laps_remain_raw)
        else:
            race_laps_remain = 0

        # Determine total laps display
        if is_time_race and estimated_total:
            total_laps_str = f"~{estimated_total}"
        elif is_time_race:
            total_laps_str = "?"
        elif isinstance(race_laps, int) and isinstance(laps_remain_raw, int):
            total_laps_str = str(race_laps + laps_remain_raw)
        else:
            total_laps_str = "?"

        pos = player.get("position", "?")
        class_pos = player.get("class_position", "?")
        flags = session.get("flags", ["Green"])
        driver_name = player.get("driver_name", "Driver")

        lines.append(
            f"Driver: {driver_name}. Race: {track_str}, Lap {race_laps}/{total_laps_str}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Race fastest lap
        race_fastest = player.get("race_fastest_lap")
        if race_fastest and race_fastest.get("time", 0) > 0:
            fast_time = format_lap_time(race_fastest["time"])
            fast_driver = race_fastest.get("driver_name", "?")
            lines.append(f"Race fastest: {fast_time} ({fast_driver})")

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
        fuel_max_pct = config.get("fuel_max_pct", 1.0)
        fuel_max_start = config.get("fuel_max_start_litres", 0)
        # Use effective max fuel (tank × restriction) for calculations.
        # If there's a fuel restriction (e.g. 40%), fuel_max_start is the
        # actual max you can start with (e.g. 22L × 0.4 = 8.8L).
        effective_fuel_max = fuel_max_start if fuel_max_start > 0 else fuel_max
        if is_fixed or incident_limit or fuel_max:
            cfg_parts = []
            if is_fixed:
                cfg_parts.append("FIXED SETUP")
            if incident_limit:
                label = (
                    "no limit"
                    if str(incident_limit).lower() in ("unlimited", "-1", "0")
                    else str(incident_limit)
                )
                cfg_parts.append(f"incidents: {label}")
            if fast_repairs:
                cfg_parts.append(f"fast repairs: {fast_repairs}")
            if fuel_max:
                if fuel_max_pct < 1.0 and fuel_max_start > 0:
                    cfg_parts.append(
                        f"tank: {fuel_max:.0f}L (max load: {fuel_max_start:.1f}L / {fuel_max_pct:.0%})"
                    )
                else:
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
                weather_str = f"Weather: {' | '.join(weather_parts)}"
                # Make it clear this is current conditions only — no forecast data
                weather_str += " (current conditions only — NO forecast data, cannot predict future weather)"
                lines.append(weather_str)

        # === Engine health ===
        oil_temp = player.get("oil_temp", 0)
        oil_press = player.get("oil_press", 0)
        water_temp = player.get("water_temp", 0)
        voltage = player.get("voltage", 0)
        engine_warnings = player.get("engine_warnings", 0)
        manifold_press = player.get("manifold_press", 0)
        engine_baseline = player.get("engine_baseline")

        # Helper: format a value with delta from baseline if available
        def _engine_part(
            label: str, value: float, unit: str, bl_key: str
        ) -> str | None:
            if value <= 0:
                return None
            base_val = (
                f"{label} {format_temp(value)}"
                if unit == "C"
                else f"{label} {value:.2f}{unit}"
            )
            if engine_baseline and bl_key in engine_baseline:
                bl = engine_baseline[bl_key]
                if bl > 0:
                    delta = value - bl
                    pct = abs(delta) / bl if bl > 0 else 0
                    if pct > 0.15:  # ±15% threshold
                        arrow = "↑" if delta > 0 else "↓"
                        delta_unit = "°C" if unit == "C" else unit
                        base_val += f" ({arrow}{abs(delta):.1f}{delta_unit} from avg)"
            return base_val

        engine_parts = []
        part = _engine_part("Oil", oil_temp, "C", "oil_temp")
        if part:
            engine_parts.append(part)
        part = _engine_part("OilP", oil_press, "bar", "oil_press")
        if part:
            engine_parts.append(part)
        part = _engine_part("Water", water_temp, "C", "water_temp")
        if part:
            engine_parts.append(part)
        part = _engine_part("Volt", voltage, "V", "voltage")
        if part:
            engine_parts.append(part)
        if manifold_press > 0 and manifold_press < 5:
            part = _engine_part("Manifold", manifold_press, "bar", "manifold_press")
            if part:
                engine_parts.append(part)
        if engine_parts:
            if engine_baseline:
                lines.append(f"Engine: {' | '.join(engine_parts)}")
            else:
                lines.append(
                    f"Engine: {' | '.join(engine_parts)} (reference ranges not available — report values only)"
                )

        if engine_warnings:
            warning_str = format_engine_warnings(engine_warnings)
            lines.append(f"  ⚠ ENGINE WARNING: {warning_str}")

        # === Player car ===
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_level = player.get("fuel_level", 0)
        fuel_est_quality = player.get("fuel_est_quality", "unreliable")
        avg_fuel_per_lap = player.get("avg_fuel_per_lap", 0)
        # Use effective max fuel (tank × restriction) for calculations
        fuel_max_pct = config.get("fuel_max_pct", 1.0)
        fuel_max_start = config.get("fuel_max_start_litres", 0)
        effective_fuel_max = fuel_max_start if fuel_max_start > 0 else fuel_max

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

        # Fuel display with qualitative descriptor and tank capacity
        # Descriptor bands: critical (<10%), low (10-20%), half (20-50%), adequate (50-80%), full (>80%)
        if fuel_pct >= 0.8:
            fuel_desc = "full"
        elif fuel_pct >= 0.5:
            fuel_desc = "adequate"
        elif fuel_pct >= 0.2:
            fuel_desc = "half"
        elif fuel_pct >= 0.1:
            fuel_desc = "low"
        else:
            fuel_desc = "⚠ CRITICAL"

        fuel_line = f"  Fuel: {fuel_level:.2f}L"
        if effective_fuel_max and fuel_level > 0:
            if fuel_max_pct < 1.0 and fuel_max_start > 0:
                fuel_line += f"/{fuel_max_start:.1f}L"
            else:
                fuel_line += f"/{fuel_max:.0f}L"
            # Show max fuel add so the LLM can't suggest more than the tank holds
            max_add = effective_fuel_max - fuel_level
            if max_add > 0:
                fuel_line += f" (max add: {max_add:.0f}L,"
            else:
                fuel_line += " ("
        else:
            fuel_line += " ("
        fuel_line += f"{format_pct(fuel_pct)}, {fuel_desc})"

        # Fuel urgency warning (shown on the same line for visibility)
        if 0 < fuel_pct < 0.2 and fuel_laps > 0:
            fuel_line += " ⚠ FUEL WARNING"

        lines.append(fuel_line)

        # Per-lap burn rate and laps remaining — shown on a separate line
        # to avoid the LLM confusing "laps remaining: unreliable" with
        # "the fuel level is unreliable". The fuel level (e.g. 15.12L, 69%)
        # IS accurate — only the laps estimate depends on having burn rate data.
        if avg_fuel_per_lap > 0 and fuel_est_quality == "good":
            lines.append(
                f"  Fuel burn: ~{avg_fuel_per_lap:.2f} L/lap (avg over recent laps)"
            )
            if fuel_laps > 0:
                lines.append(
                    f"  Fuel range: ~{fuel_laps:.1f} laps at current burn rate"
                )
                # Critical: fuel range vs race laps remaining comparison
                if race_laps_remain > 0:
                    if fuel_laps < race_laps_remain:
                        deficit_laps = race_laps_remain - fuel_laps
                        lines.append(
                            f"  ⚠ FUEL SHORTAGE: ~{deficit_laps:.1f} laps short — cannot finish without pit stop"
                        )
                    elif fuel_laps < race_laps_remain + 1:
                        # Within 1 lap of shortage — margins are too thin to be safe
                        margin = fuel_laps - race_laps_remain
                        if margin < 0.3:
                            lines.append(
                                f"  ⚠ FUEL TIGHT: ~{margin:.1f} laps margin — will NOT finish if burn rate varies. Pit recommended."
                            )
                        else:
                            lines.append(
                                f"  ⚠ FUEL TIGHT: only ~{margin:.1f} laps margin — pit if safety car or incident possible"
                            )
        elif avg_fuel_per_lap > 0 and fuel_est_quality == "rough":
            lines.append(
                f"  Fuel burn: ~{avg_fuel_per_lap:.2f} L/lap (approx, based on 1-2 laps)"
            )
            if fuel_laps > 0:
                lines.append(f"  Fuel range: ~{fuel_laps:.1f} laps (approx)")
                if race_laps_remain > 0:
                    if fuel_laps < race_laps_remain:
                        deficit_laps = race_laps_remain - fuel_laps
                        lines.append(
                            f"  ⚠ FUEL SHORTAGE: ~{deficit_laps:.1f} laps short (approx) — cannot finish without pit stop"
                        )
                    elif fuel_laps < race_laps_remain + 1:
                        margin = fuel_laps - race_laps_remain
                        if margin < 0.3:
                            lines.append(
                                f"  ⚠ FUEL TIGHT: ~{margin:.1f} laps margin (approx) — will NOT finish if burn rate varies. Pit recommended."
                            )
                        else:
                            lines.append(
                                f"  ⚠ FUEL TIGHT: only ~{margin:.1f} laps margin (approx) — pit if safety car or incident possible"
                            )
        else:
            # No burn rate data yet — make it clear the LLM must NOT invent
            # a specific L/lap figure. "calculating" was misread as "almost ready"
            # by the LLM, causing it to fabricate burn rates like "1.3L/lap".
            lines.append("  Fuel burn: unknown — per-lap rate not yet available")

        # Fuel recommendation — calculate how much fuel to add at next pit stop.
        # Do the math here so the LLM doesn't invent wrong amounts.
        # Only shown when we have burn rate data AND know how many laps are left.
        if (
            avg_fuel_per_lap > 0
            and race_laps_remain > 0
            and effective_fuel_max > 0
            and fuel_level > 0
        ):
            fuel_needed = avg_fuel_per_lap * race_laps_remain
            fuel_deficit = fuel_needed - fuel_level
            if fuel_deficit > 0:
                # Add 1-lap safety margin to avoid running dry on the last lap,
                # then round up to whole litres (add_fuel only accepts integers).
                add_litres = min(
                    int(fuel_deficit + avg_fuel_per_lap)
                    + 1,  # deficit + 1 lap margin, ceil
                    int(effective_fuel_max - fuel_level),  # max fuel capacity
                )
                add_litres = max(1, add_litres)  # at least 1L
                quality_note = "" if fuel_est_quality == "good" else " (approx)"
                lines.append(
                    f"  Fuel to finish: ~{fuel_needed:.1f}L for {race_laps_remain} laps{quality_note}"
                )
                lines.append(f"  Fuel to add: {add_litres}L")

        # Race duration and time remaining (for time-based races)
        # Show both time and laps — the LLM needs to know the race is e.g. 30 min
        # so it can reason about fuel needs even without lap-by-lap history.
        session_time = session.get("session_time", 0)
        time_remain_sec = session.get("time_remain", 0)
        if is_time_race:
            # Time-based race (e.g. 30 min sprint)
            if session_time > 0:
                session_min = int(session_time / 60)
                lines.append(f"  Race duration: {session_min} min")
            if time_remain_sec > 0:
                remain_min = time_remain_sec / 60
                if remain_min < 1:
                    lines.append(f"  Time remaining: {int(time_remain_sec)}s")
                else:
                    lines.append(f"  Time remaining: ~{remain_min:.1f} min")
            if race_laps_remain > 0:
                if race_laps_remain <= 1:
                    lines.append(
                        f"  Race laps remaining: ~{race_laps_remain} — RACE ENDING SOON"
                    )
                    lines.append(
                        "  ⚠ Pit stop costs ~30s+ — do not pit unless car cannot finish"
                    )
                elif race_laps_remain <= 3:
                    lines.append(
                        f"  Race laps remaining: ~{race_laps_remain} laps (estimated) — RACE ENDING SOON"
                    )
                    lines.append(
                        "  ⚠ Pit stop costs ~30s+ — only pit if absolutely necessary (e.g. fuel shortage)"
                    )
                else:
                    lines.append(
                        f"  Race laps remaining: ~{race_laps_remain} laps (estimated)"
                    )

        # Tyres — show staleness indicator
        # iRacing freezes tyre data on track for most cars — only updates in pits.
        # Always show staleness so the LLM knows whether to trust the values.
        tyres = player.get("tyres", {})
        tyre_odometers = player.get("tyre_odometers", {})
        tyre_staleness = player.get("tyre_staleness", "unknown")
        tyre_baseline = player.get("tyre_baseline")

        # Helper: format tyre temp with delta from baseline
        def _tyre_temp_str(corner: str, temp: float) -> str:
            base = format_temp(temp)
            if tyre_baseline and corner in tyre_baseline and temp > 0:
                bl = tyre_baseline[corner]
                if bl > 0:
                    delta = temp - bl
                    pct = abs(delta) / bl if bl > 0 else 0
                    if pct > 0.15:  # ±15% threshold
                        arrow = "↑" if delta > 0 else "↓"
                        base += f" ({arrow}{abs(delta):.1f}°C from avg)"
            return base

        if tyres:
            is_stale = tyre_staleness != "live"
            if tyre_staleness == "stale":
                lines.append("  Tyres: unreliable (on track, data not updating)")
            elif tyre_staleness == "live":
                lines.append("  Tyres: (data updating — fresh from pit stop)")
            else:
                lines.append("  Tyres: data availability not yet confirmed")

            # When tyre data is stale/frozen, check if all pressures are identical
            # and collapse them into one line to avoid the LLM saying "pressures stable"
            pressures = []
            for corner in ["LF", "RF", "LR", "RR"]:
                ts = tyres.get(corner, {})
                p = ts.get("cold_pressure", 0)
                if p > 0:
                    pressures.append(p)

            all_pressures_same = (
                len(pressures) == 4 and len(set(f"{p:.2f}" for p in pressures)) == 1
            )

            if is_stale and all_pressures_same and pressures:
                # Collapse identical stale pressures into one line
                # Pressures are stored in kPa; convert to PSI for display
                stale_label = (
                    "not updating"
                    if tyre_staleness == "stale"
                    else "may not be updating"
                )
                lines.append(
                    f"    All pressures: {kpa_to_psi(pressures[0]):.1f}PSI ({stale_label} — normal for iRacing cars on track)"
                )
                # Show per-corner data without pressure
                for corner in ["LF", "RF", "LR", "RR"]:
                    ts = tyres.get(corner, {})
                    if not ts:
                        continue
                    avg_temp = ts.get("temp_center", 0)
                    wear_center = ts.get("wear_center", 0)
                    odo = tyre_odometers.get(corner, 0)
                    parts = [_tyre_temp_str(corner, avg_temp)]
                    if wear_center > 0:
                        parts.append(f"{format_wear(wear_center)}")
                    if odo > 0:
                        parts.append(f"{odo:.0f}km")
                    lines.append(f"    {CORNER_NAMES[corner]}: {' | '.join(parts)}")
            else:
                # Show full per-corner data including pressure
                for corner in ["LF", "RF", "LR", "RR"]:
                    ts = tyres.get(corner, {})
                    if not ts:
                        continue
                    avg_temp = ts.get("temp_center", 0)
                    pressure = ts.get("cold_pressure", 0)
                    wear_center = ts.get("wear_center", 0)
                    odo = tyre_odometers.get(corner, 0)
                    parts = [_tyre_temp_str(corner, avg_temp)]
                    if pressure > 0:
                        # Pressure stored in kPa; convert to PSI for display
                        parts.append(f"{kpa_to_psi(pressure):.1f}PSI")
                    if wear_center > 0:
                        parts.append(f"{format_wear(wear_center)}")
                    if odo > 0:
                        parts.append(f"{odo:.0f}km")
                    lines.append(f"    {CORNER_NAMES[corner]}: {' | '.join(parts)}")

        # Brakes + brake bias
        # Brake line pressures are stored in kPa; convert to PSI for display
        brake_pressures = player.get("brake_pressures", {})
        brake_temps = []
        for corner in ["LF", "RF", "LR", "RR"]:
            bp = brake_pressures.get(corner, 0)
            if bp > 0:
                # Convert kPa to PSI for display
                brake_temps.append(f"{CORNER_NAMES[corner]} {kpa_to_psi(bp):.0f}PSI")
        if brake_temps:
            lines.append(f"  Brakes: {' / '.join(brake_temps)}")
        # Brake bias: stored as percentage (0-100 range after sanitisation)
        brake_bias = player.get("brake_bias", 0)
        if brake_bias and 10 < brake_bias < 90:
            # Only show if value is plausible (10-90% range)
            lines.append(f"  Brake bias: {brake_bias:.1f}%")

        # Incidents (safety-rating points, NOT car damage) / penalties
        incidents = player.get("incidents", 0)
        team_incidents = player.get("team_incidents", 0)
        weight_penalty = player.get("weight_penalty", 0)
        fast_repairs_used = player.get("fast_repairs_used", 0)
        repair_time = player.get("pit_repair_time_left", 0)
        opt_repair_time = player.get("pit_opt_repair_time_left", 0)

        incident_parts = []
        if incidents > 0 or team_incidents > 0:
            incident_parts.append(
                f"⚠ {incidents}x incidents (team {team_incidents}x) — safety-rating points, not car damage"
            )
        damage_parts = []
        if weight_penalty > 0:
            damage_parts.append(f"+{weight_penalty:.2f}kg damage weight")
        if fast_repairs_used > 0:
            damage_parts.append(f"{fast_repairs_used} fast repairs used")
        if repair_time > 0:
            damage_parts.append(f"{repair_time:.0f}s repair time needed")
        if opt_repair_time > 0:
            damage_parts.append(f"{opt_repair_time:.0f}s opt repair time")
        if incident_parts:
            lines.append(f"  Incidents: {' | '.join(incident_parts)}")
        if damage_parts:
            lines.append(f"  Damage: {' | '.join(damage_parts)}")
        elif not weight_penalty and not repair_time and not opt_repair_time:
            # Only show this when there's no damage data at all — prevents the
            # LLM from saying "no damage reported" when it simply doesn't have data
            lines.append(
                "  Body/aero damage: not available (only incident points tracked)"
            )

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

        # Player lap times — shown even without lap history so the LLM
        # knows the driver's own pace (not just nearby cars' times)
        last_lap = player.get("last_lap_time", 0)
        best_lap = player.get("best_lap_time", 0)
        lap_time_parts = []
        if last_lap > 0:
            lap_time_parts.append(f"last {format_lap_time(last_lap)}")
        if best_lap > 0:
            lap_time_parts.append(f"best {format_lap_time(best_lap)}")
        if lap_time_parts:
            lines.append(f"  Pace: {' | '.join(lap_time_parts)}")

        # Proximity — CarDistAhead/Behind from iRacing are in metres, NOT seconds.
        # Use the nearby cars' gap_seconds (computed from track position) for
        # meaningful time gaps, and show iRacing's distance values as metres.
        car_prox = player.get("car_left_right", 0)
        dist_ahead = player.get("car_dist_ahead", 0)
        dist_behind = player.get("car_dist_behind", 0)
        tow_time = player.get("tow_time", 0)
        PROXIMITY_SENTINEL = 10000.0  # Values above this mean "no car nearby"
        prox_parts = []
        if car_prox:
            prox_str = format_car_proximity(car_prox)
            if prox_str:
                prox_parts.append(prox_str)
        if 0 < dist_ahead < PROXIMITY_SENTINEL:
            # Convert metres to approximate seconds using best lap time if available
            # so the LLM can reason about gaps. Fall back to raw metres if no time.
            best_lap = player.get("best_lap_time", 0)
            track_length_km = config.get("track_length_km", 0)
            if best_lap > 0 and track_length_km > 0:
                avg_speed_mps = (track_length_km * 1000) / best_lap
                gap_seconds = dist_ahead / avg_speed_mps if avg_speed_mps > 0 else 0
                if gap_seconds > 0:
                    prox_parts.append(
                        f"nearest ahead: +{gap_seconds:.1f}s ({dist_ahead:.0f}m)"
                    )
                else:
                    prox_parts.append(f"nearest ahead: {dist_ahead:.0f}m")
            else:
                prox_parts.append(f"nearest ahead: {dist_ahead:.0f}m")
        if 0 < dist_behind < PROXIMITY_SENTINEL:
            best_lap = player.get("best_lap_time", 0)
            track_length_km = config.get("track_length_km", 0)
            if best_lap > 0 and track_length_km > 0:
                avg_speed_mps = (track_length_km * 1000) / best_lap
                gap_seconds = dist_behind / avg_speed_mps if avg_speed_mps > 0 else 0
                if gap_seconds > 0:
                    prox_parts.append(
                        f"nearest behind: -{gap_seconds:.1f}s ({dist_behind:.0f}m)"
                    )
                else:
                    prox_parts.append(f"nearest behind: {dist_behind:.0f}m")
            else:
                prox_parts.append(f"nearest behind: {dist_behind:.0f}m")
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
        # Driver name aliases come from config (parsed from teammates list)
        # and from the race state (populated by iracing_client.detect_team_indices)
        driver_aliases = self.config.get("driver_aliases", {})
        # Also merge any aliases passed in the state (from RaceState.set_team_indices)
        state_aliases = state.get("driver_aliases", {})
        if state_aliases:
            merged_aliases = dict(driver_aliases)
            merged_aliases.update(state_aliases)
            driver_aliases = merged_aliases

        if nearby:
            lines.append("\nNearby:")
            for car in nearby[: self.include_nearby_cars]:
                name = car.get("driver_name", f"P{car.get('position', '?')}")
                pos = car.get("position", "?")
                gap = car.get("gap_seconds", 0)
                lap_time = car.get("last_lap_time", -1)
                p2p = "P2P available" if car.get("p2p_available") else ""
                on_pit = " (in pits)" if car.get("on_pit_road") else ""

                # Resolve display name: show "Nickname / Real Name" when alias exists
                display_name = name
                if name in driver_aliases and driver_aliases[name] != name:
                    display_name = f"{name} / {driver_aliases[name]}"
                elif (
                    car.get("is_teammate")
                    and car.get("real_name")
                    and car["real_name"] != name
                ):
                    display_name = f"{name} / {car['real_name']}"

                car_str = f"  P{pos} ({display_name}): {format_gap(gap)}"
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
