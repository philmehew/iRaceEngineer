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
    """Format a temperature value."""
    if temp <= 0:
        return "N/A"
    return f"{temp:.0f}°C"


def format_pct(value: float) -> str:
    """Format a percentage (0-1) as a whole number percent."""
    if value <= 0:
        return "N/A"
    return f"{value * 100:.0f}%"


def format_wear(wear: float) -> str:
    """Format tyre wear (0-1) as a percentage."""
    if wear <= 0:
        return "N/A"
    return f"{wear * 100:.1f}%"


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
            "You are a race engineer for a sim racing team. Give concise, actionable "
            "strategy advice based on the race data provided. Use driver names when "
            "referring to teammates. Be direct — the driver is in a race and needs "
            "quick answers. If you're unsure, say so. Never invent data not provided.\n\n"
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

        # Session line
        track = session.get("track_name", "Unknown")
        laps_remain = session.get("laps_remain", "?")
        race_laps = session.get("race_laps", "?")
        pos = player.get("position", "?")

        lines.append(
            f"Race: {track}, P{pos}, Lap {race_laps}/{race_laps + laps_remain if isinstance(laps_remain, int) and isinstance(race_laps, int) else '?'}"
        )

        # Fuel
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        lines.append(f"Fuel: {format_pct(fuel_pct)} (~{fuel_laps:.0f} laps remaining)")
        lines.append(f"Laps remaining: {laps_remain}")

        return "\n".join(lines)

    def _build_medium(self, state: dict) -> str:
        """Build medium context: minimal + gaps, tyre temps, flags."""
        lines = []
        session = state.get("session", {})
        player = state.get("player", {})

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
            f"Race: {track_str}, Lap {race_laps}/{total_laps}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Player car
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_level = player.get("fuel_level", 0)
        lines.append("\nYour car:")
        lines.append(
            f"  Fuel: {fuel_level:.1f}L ({format_pct(fuel_pct)}), ~{fuel_laps:.1f} laps remaining"
        )

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

        return "\n".join(lines)

    def _build_full(self, state: dict) -> str:
        """Build full context: medium + lap trends, nearby cars, pit window, weather, damage."""
        lines = []
        session = state.get("session", {})
        player = state.get("player", {})

        # === Session header ===
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
            f"Race: {track_str}, Lap {race_laps}/{total_laps}, P{pos} (Class P{class_pos})"
        )
        lines.append(f"Flags: {', '.join(flags)}")

        # Weather
        weather = session.get("weather", {})
        if weather:
            track_temp = weather.get("track_temp", 0)
            air_temp = weather.get("air_temp", 0)
            wetness = weather.get("precipitation", 0)
            wet_str = f", Wetness: {wetness:.0%}" if wetness > 0 else ""
            if track_temp > 0 or air_temp > 0:
                lines.append(
                    f"Track: {format_temp(track_temp)}, Air: {format_temp(air_temp)}{wet_str}"
                )

        # === Player car ===
        fuel_laps = player.get("fuel_laps_remaining", 0)
        fuel_pct = player.get("fuel_pct", 0)
        fuel_level = player.get("fuel_level", 0)
        fuel_rate = player.get("fuel_use_per_hour", 0)

        lines.append("\nYour car:")
        lines.append(f"  Position: P{pos} (Class P{class_pos})")
        lines.append(
            f"  Fuel: {fuel_level:.1f}L ({format_pct(fuel_pct)}), ~{fuel_laps:.1f} laps remaining"
        )
        if fuel_rate > 0:
            lines.append(f"  Fuel burn rate: {fuel_rate:.1f} L/hr")

        # Tyres
        tyres = player.get("tyres", {})
        if tyres:
            for corner in ["LF", "RF", "LR", "RR"]:
                ts = tyres.get(corner, {})
                if not ts:
                    continue
                avg_temp = ts.get("temp_center", 0)
                pressure = ts.get("cold_pressure", 0)
                wear_center = ts.get("wear_center", 0)
                parts = [f"{format_temp(avg_temp)}"]
                if pressure > 0:
                    parts.append(f"{pressure:.1f} PSI")
                if wear_center > 0:
                    parts.append(f"wear {format_wear(wear_center)}")
                lines.append(f"  {corner}: {' | '.join(parts)}")

        # Brakes
        brake_pressures = player.get("brake_pressures", {})
        brake_temps = []
        for corner in ["LF", "RF", "LR", "RR"]:
            bp = brake_pressures.get(corner, 0)
            if bp > 0:
                brake_temps.append(f"{corner} {bp:.0f}")
        if brake_temps:
            lines.append(f"  Brakes: {' / '.join(brake_temps)}")

        # Damage / incidents
        incidents = player.get("incidents", 0)
        team_incidents = player.get("team_incidents", 0)
        if incidents > 0 or team_incidents > 0:
            lines.append(f"  Incidents: {incidents} (team: {team_incidents})")

        # Push-to-pass
        p2p_remaining = player.get("p2p_remaining", 0)
        p2p_active = player.get("p2p_active", False)
        if p2p_remaining > 0:
            lines.append(
                f"  Push-to-pass: {p2p_remaining} remaining{' (ACTIVE)' if p2p_active else ''}"
            )

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
