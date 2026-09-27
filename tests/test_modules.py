"""Unit tests for iRaceEngineer modules."""

from race_state import (
    RaceState,
    DriverState,
    TyreState,
    LapRecord,
    SectorTimeTracker,
    FLAG_GREEN,
    FLAG_YELLOW,
)
from context_builder import (
    ContextBuilder,
    TRACK_WETNESS_LABELS,
    format_lap_time,
    format_gap,
    format_temp,
    format_pct,
    format_engine_warnings,
    format_car_proximity,
)
from capture import TelemetryReplay


# --- Fixtures ---


def sample_telemetry(lap=10, fuel=72.0, flags=4):
    """Generate sample telemetry data for testing."""
    return {
        "SessionFlags": flags,
        "CarIdxSessionFlags": [0, 0, 0, 0],
        "SessionLapsRemain": 60 - lap,
        "SessionTimeRemain": 3600 - lap * 90,
        "SessionNum": 0,
        "SessionState": 4,
        "RaceLaps": lap,
        "PlayerCarIdx": 0,
        "PlayerCarPosition": 4,
        "PlayerCarClassPosition": 2,
        "Lap": lap,
        "LapCompleted": lap - 1,
        "LapDistPct": 0.5,
        "Speed": 85.0,
        "RPM": 8200,
        "Gear": 5,
        "Throttle": 0.95,
        "Brake": 0.0,
        "FuelLevel": fuel,
        "FuelLevelPct": fuel / 110,
        "FuelUsePerHour": 28.5,
        "LapCurrentLapTime": 46.0,
        "LapBestLapTime": 92.5,
        "LapLastLapTime": 93.5,
        "LapDeltaToBestLap": 1.0,
        "LFtempCL": 95.0,
        "LFtempCM": 97.0,
        "LFtempCR": 100.0,
        "RFtempCL": 96.0,
        "RFtempCM": 99.0,
        "RFtempCR": 102.0,
        "LRtempCL": 102.0,
        "LRtempCM": 105.0,
        "LRtempCR": 109.0,
        "RRtempCL": 103.0,
        "RRtempCM": 107.0,
        "RRtempCR": 111.0,
        "LFcoldPressure": 26.5,
        "RFcoldPressure": 26.3,
        "LRcoldPressure": 24.8,
        "RRcoldPressure": 24.5,
        "LFwearL": 0.08,
        "LFwearM": 0.10,
        "LFwearR": 0.12,
        "RFwearL": 0.07,
        "RFwearM": 0.09,
        "RFwearR": 0.11,
        "LRwearL": 0.10,
        "LRwearM": 0.14,
        "LRwearR": 0.16,
        "RRwearL": 0.11,
        "RRwearM": 0.15,
        "RRwearR": 0.18,
        "TrackTemp": 27.0,
        "TrackWetness": 0.0,
        "AirTemp": 22.0,
        "AirPressure": 1013.0,
        "Precipitation": 0.0,
        "WindDir": 180.0,
        "WindVel": 3.5,
        "OnPitRoad": False,
        "PitstopActive": False,
        "PitsOpen": True,
        "FastRepairAvailable": True,
        "TireSetsAvailable": 3,
        "TireSetsUsed": 1,
        "PlayerTireCompound": 0,
        "P2P_Status": 0,
        "P2P_Count": 3,
        "CarIdxPosition": [1, 2, 3, 4],
        "CarIdxClassPosition": [1, 1, 2, 2],
        "CarIdxLap": [lap, lap, lap, lap],
        "CarIdxLapDistPct": [0.52, 0.50, 0.48, 0.46],
        "CarIdxOnPitRoad": [False, False, False, False],
        "CarIdxBestLapTime": [91.8, 92.0, 92.3, 92.5],
        "CarIdxLastLapTime": [91.9, 92.1, 92.4, 93.5],
        "CarIdxP2P_Status": [0, 0, 1, 0],
        "CarIdxP2P_Count": [2, 4, 1, 3],
        "CarIdxTireCompound": [0, 0, 1, 0],
        "CarIdxTrackSurface": [3, 3, 3, 3],  # 3 = on track (CarIdxTrackSurface enum)
        "SessionTime": 900.0 + lap * 90,
        "PlayerCarMyIncidentCount": 0,
        "PlayerCarTeamIncidentCount": 2,
        "CarDistAhead": 2.1,
        "CarDistBehind": -1.8,
        # Engine health
        "OilTemp": 95.0,
        "OilPress": 4.2,
        "OilLevel": 6.5,
        "WaterTemp": 88.0,
        "WaterLevel": 6.7,
        "FuelPress": 3.9,
        "EngineWarnings": 0,
        "ManifoldPress": 1.02,
        "Voltage": 13.8,
        # Car status
        "IsOnTrack": True,
        "IsInGarage": False,
        # Damage and penalties
        "PlayerCarWeightPenalty": 0.0,
        "PlayerFastRepairsUsed": 0,
        "PitRepairLeft": 0.0,
        "PitOptRepairLeft": 0.0,
        # Proximity
        "CarLeftRight": 0,
        "PlayerCarTowTime": 0.0,
        # G-forces
        "LatAccel": 1.2,
        "LongAccel": 0.3,
        "VertAccel": 9.8,
        # Brake bias
        "dcBrakeBias": 54.0,
        # Shift lights
        "ShiftIndicatorPct": 0.75,
        "PlayerCarSLShiftRPM": 6800.0,
        # Tyre odometers
        "LFodometer": 1500.0,
        "RFodometer": 1500.0,
        "LRodometer": 1500.0,
        "RRodometer": 1500.0,
        # Track conditions
        "WeatherDeclaredWet": False,
        "PlayerTrackSurface": 3,
        "PlayerTrackSurfaceMaterial": 1,
    }


def sample_session_info():
    """Generate sample session info for testing."""
    return {
        "WeekendInfo": {
            "TrackName": "Circuit de Spa-Francorchamps",
            "TrackConfigName": "Grand Prix",
            "TrackLength": "7.004 km",
            "TrackNumTurns": 20,
            "TrackPitSpeedLimit": "60.00 kph",
            "MaxDrivers": 1,
            "WeekendOptions": {
                "IsFixedSetup": 0,
                "IncidentLimit": "17x",
                "FastRepairsLimit": "2",
                "NumStarters": 30,
            },
        },
        "SessionInfo": {
            "Sessions": [
                {"SessionNum": 0, "SessionType": "Race", "SessionName": "Race"}
            ]
        },
        "DriverInfo": {
            "DriverCarFuelMaxLtr": 110.0,
            "DriverCarIdleRPM": 800.0,
            "DriverCarRedLine": 7500.0,
            "DriverCarSLShiftRPM": 6800.0,
            "DriverCarSLFirstRPM": 5500.0,
            "DriverCarSLLastRPM": 7200.0,
            "DriverCarSLBlinkRPM": 7000.0,
            "DriverCarEstLapTime": 92.5,
            "Drivers": [
                {
                    "CarIdx": 0,
                    "UserName": "Patrik Farsang",
                    "CarNumber": "7",
                    "TeamName": "iRaceEngineer",
                    "CurDriverIncidentCount": 0,
                },
                {
                    "CarIdx": 1,
                    "UserName": "Wayne Smith8",
                    "CarNumber": "4",
                    "TeamName": "iRaceEngineer",
                    "CurDriverIncidentCount": 2,
                },
            ],
        },
        "SplitTimeInfo": {
            "Sectors": [
                {"SectorNum": 1, "SectorStartPct": 0.0},
                {"SectorNum": 2, "SectorStartPct": 0.3333},
                {"SectorNum": 3, "SectorStartPct": 0.6667},
            ]
        },
    }


def make_state(lap=10, fuel=72.0, flags=4):
    """Create a RaceState with sample data."""
    config = {
        "prompt": {
            "context_depth": "full",
            "include_lap_history": 5,
            "include_nearby_cars": 3,
        }
    }
    state = RaceState(config)
    state.update(
        sample_telemetry(lap=lap, fuel=fuel, flags=flags),
        sample_session_info(),
        {0: "Patrik Farsang", 1: "Wayne Smith8", 2: "Andre Groove", 3: "Dan Golden"},
    )
    return state


# --- RaceState tests ---


class TestRaceState:
    def test_update_populates_player(self):
        state = make_state()
        assert state.player.position == 4
        assert state.player.lap == 10
        assert state.player.fuel_level == 72.0

    def test_update_populates_session(self):
        state = make_state()
        assert state.session.track_name == "Circuit de Spa-Francorchamps"
        assert state.session.laps_remain == 50  # 60 - 10

    def test_update_populates_tyres(self):
        state = make_state()
        assert "LF" in state.player.tyres
        assert state.player.tyres["LF"].temp_center == 97.0
        assert state.player.tyres["RR"].temp_center == 107.0

    def test_update_populates_nearby_cars(self):
        state = make_state()
        assert len(state.nearby_cars) > 0
        assert state.nearby_cars[0].driver_name is not None

    def test_flags_list_green(self):
        state = make_state(flags=FLAG_GREEN)
        assert "Green" in state.flags_list

    def test_flags_list_yellow(self):
        state = make_state(flags=FLAG_YELLOW)
        assert "Yellow" in state.flags_list

    def test_flags_list_combined(self):
        state = make_state(flags=FLAG_GREEN | FLAG_YELLOW)
        flags = state.flags_list
        assert "Green" in flags
        assert "Yellow" in flags

    def test_snapshot_includes_session(self):
        state = make_state()
        snap = state.get_snapshot()
        assert "session" in snap
        assert snap["session"]["track_name"] == "Circuit de Spa-Francorchamps"

    def test_snapshot_includes_player(self):
        state = make_state()
        snap = state.get_snapshot()
        assert "player" in snap
        assert snap["player"]["position"] == 4
        assert "tyres" in snap["player"]

    def test_snapshot_includes_nearby_cars(self):
        state = make_state()
        snap = state.get_snapshot()
        assert "nearby_cars" in snap
        assert len(snap["nearby_cars"]) > 0

    def test_snapshot_includes_lap_history(self):
        state = make_state()
        snap = state.get_snapshot()
        assert "lap_history" in snap

    def test_fuel_laps_remaining_with_history(self):
        state = make_state()
        # With no lap history, fuel_laps_remaining falls back to
        # burn rate + estimated lap time from session info
        # (DriverCarEstLapTime=92.5, FuelUsePerHour=28.5, FuelLevel=72.0)
        # fuel_per_lap = 28.5 * (92.5 / 3600) ≈ 0.73 L/lap
        # laps_remaining = 72.0 / 0.73 ≈ 98.3
        assert state.fuel_laps_remaining > 0

        # Simulate a completed lap by adding history manually
        state.player.lap_history.append(
            LapRecord(
                lap_number=1,
                lap_time=92.5,
                fuel_used=3.8,
                fuel_at_start=110.0,
                fuel_at_end=106.2,
            )
        )
        state.player.lap_history.append(
            LapRecord(
                lap_number=2,
                lap_time=93.0,
                fuel_used=3.8,
                fuel_at_start=106.2,
                fuel_at_end=102.4,
            )
        )
        # Now fuel_laps_remaining should use actual lap history
        laps = state.fuel_laps_remaining
        assert laps > 0

    def test_driver_names_set(self):
        state = make_state()
        assert state.player.driver_name == "Patrik Farsang"


# --- ContextBuilder tests ---


class TestContextBuilder:
    def test_minimal_context(self):
        state = make_state()
        config = {"prompt": {"context_depth": "minimal", "system": "test"}}
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert "Fuel" in content
        assert len(content) < 300  # Minimal should be short

    def test_medium_context(self):
        state = make_state()
        config = {"prompt": {"context_depth": "medium", "system": "test"}}
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        assert "Tyres" in content or "tyre" in content.lower()

    def test_full_context(self):
        state = make_state()
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        assert "Nearby" in content
        # Trend is only shown when there are 3+ lap records in history
        # Single update test won't have lap history, so check for other full-context fields
        assert "Pit" in content or "Fuel" in content  # Full context includes pit status

    def test_context_with_question(self):
        state = make_state()
        config = {"prompt": {"context_depth": "full", "system": "test"}}
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot(), question="Should I pit?")
        last_msg = messages[-1]["content"]
        assert "Should I pit?" in last_msg

    def test_qa_history_injected_into_prompt(self):
        state = make_state()
        config = {"prompt": {"context_depth": "minimal", "system": "test"}}
        builder = ContextBuilder(config)
        builder.qa_history.append("Should I pit?", "Box in 2 laps.")
        messages = builder.build_prompt(
            state.get_snapshot(), question="What if it rains?"
        )
        # system, history user, history assistant, snapshot+question
        assert len(messages) == 4
        assert messages[1] == {"role": "user", "content": "Should I pit?"}
        assert messages[2] == {"role": "assistant", "content": "Box in 2 laps."}
        assert "What if it rains?" in messages[3]["content"]

    def test_qa_history_respects_max_exchanges(self):
        config = {
            "prompt": {
                "context_depth": "minimal",
                "system": "test",
                "qa_history_max": 2,
            }
        }
        builder = ContextBuilder(config)
        for i in range(5):
            builder.qa_history.append(f"Q{i}", f"A{i}")
        messages = builder.qa_history.as_messages()
        # 2 exchanges = 4 messages, and only the newest survive
        assert len(messages) == 4
        assert messages[0]["content"] == "Q3"
        assert messages[-1]["content"] == "A4"

    def test_qa_history_disabled(self):
        config = {
            "prompt": {
                "context_depth": "minimal",
                "system": "test",
                "qa_history_max": 0,
            }
        }
        builder = ContextBuilder(config)
        builder.qa_history.append("Q", "A")
        assert builder.qa_history.as_messages() == []
        messages = builder.build_prompt(state=make_state().get_snapshot())
        assert len(messages) == 2  # stateless — system + user only

    def test_qa_history_thread_safety(self):
        # Concurrent appends must never lose the lock or corrupt the list
        import threading

        config = {
            "prompt": {
                "context_depth": "minimal",
                "system": "test",
                "qa_history_max": 100,
            }
        }
        builder = ContextBuilder(config)
        errors = []

        def worker(n):
            try:
                for i in range(200):
                    builder.qa_history.append(f"Q{n}-{i}", f"A{n}-{i}")
                    builder.qa_history.as_messages()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # Cap is 100 exchanges = 200 messages; reaching it intact means no
        # appends were lost or corrupted under concurrency
        assert len(builder.qa_history.as_messages()) == 200

    def test_format_lap_time(self):
        assert format_lap_time(92.5) == "1:32.500"
        assert format_lap_time(-1) == "N/A"
        assert format_lap_time(0) == "N/A"

    def test_format_gap(self):
        assert format_gap(2.1) == "+2.100s"
        assert format_gap(-1.8) == "-1.800s"
        assert format_gap(0) == "0.000s"

    def test_format_temp(self):
        assert format_temp(97.0) == "97.0°C"
        assert format_temp(0) == "N/A"

    def test_format_pct(self):
        assert format_pct(0.65) == "65%"

    def test_format_engine_warnings(self):
        assert format_engine_warnings(0) == ""
        assert format_engine_warnings(1) == "water temp"
        assert format_engine_warnings(4) == "oil pressure"
        assert format_engine_warnings(0x20) == "rev limiter"
        # Combined warnings: 0x05 = water temp | oil pressure
        result = format_engine_warnings(0x05)
        assert "water temp" in result
        assert "oil pressure" in result
        assert "water temp" in result or "fuel pressure" in result

    def test_format_car_proximity(self):
        assert format_car_proximity(0) == ""  # off
        assert format_car_proximity(1) == ""  # clear (no cars)
        assert format_car_proximity(2) == "car LEFT"
        assert format_car_proximity(3) == "car RIGHT"
        assert "LEFT" in format_car_proximity(4) and "RIGHT" in format_car_proximity(
            4
        )  # both

    def test_full_context_engine_health_normal_suppressed(self):
        """Normal engine values (no anomaly, no warning) should be suppressed."""
        state = make_state()
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        # Normal engine data should NOT appear — no anomaly or warning
        assert "Engine:" not in content

    def test_full_context_engine_health_warning_shown(self):
        """Engine data should appear when there's an engine warning."""
        state = make_state()
        # Set an engine warning (fuel pressure = bit 0x02)
        state.player.engine_warnings = 0x02
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        # Engine data should appear when there's a warning
        assert "Engine:" in content
        assert "Engine warning" in content

    def test_engine_warning_in_action_line_with_fuel_action(self):
        """Engine warning should be appended to Action line when there's also a fuel action."""
        state = make_state()
        # Set engine warning (fuel pressure = bit 0x02)
        state.player.engine_warnings = 0x02
        # Set critical fuel to trigger Pit action
        state.player.fuel_level = 0.9
        state.player.fuel_pct = 0.04
        state.player.fuel_laps_remaining = 0.3
        state.player.avg_fuel_per_lap = 1.32
        state.player.fuel_est_quality = "good"
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        # Action line should contain short engine issue suffix
        # (no "Action:" prefix — just the advice text)
        assert "engine issue" in content

    def test_engine_warning_in_action_line_standalone(self):
        """Engine warning alone should create a standalone Action line when no fuel action."""
        state = make_state()
        # Set engine warning (oil pressure = bit 0x04)
        state.player.engine_warnings = 0x04
        # Adequate fuel — no Pit action
        state.player.fuel_pct = 0.50
        state.player.fuel_level = 11.0
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        # Action line should exist with short engine warning
        # (no "Action:" prefix — just the advice text)
        assert "Engine issue" in content
        assert "check engine" in content

    def test_full_context_includes_session_config(self):
        state = make_state()
        config = {
            "prompt": {
                "context_depth": "full",
                "system": "test",
                "include_lap_history": 5,
                "include_nearby_cars": 3,
            }
        }
        builder = ContextBuilder(config)
        messages = builder.build_prompt(state.get_snapshot())
        content = messages[1]["content"]
        # Session config should appear in full context
        assert "incidents:" in content
        assert "tank:" in content

    def test_snapshot_includes_engine_health(self):
        state = make_state()
        snap = state.get_snapshot()
        player = snap["player"]
        assert player["oil_temp"] == 95.0
        assert player["oil_press"] == 4.2
        assert player["water_temp"] == 88.0
        assert player["engine_warnings"] == 0
        assert player["voltage"] == 13.8
        assert player["brake_bias"] == 54.0
        assert player["is_on_track"] is True
        assert player["is_in_garage"] is False

    def test_snapshot_includes_session_config(self):
        state = make_state()
        snap = state.get_snapshot()
        config = snap["session"]["config"]
        assert config["fuel_max_litres"] == 110.0
        assert config["track_num_turns"] == 20
        assert config["track_length_km"] == 7.004
        assert config["is_fixed_setup"] is False
        assert config["incident_limit"] == "17x"
        assert config["fast_repairs_limit"] == "2"
        assert config["shift_rpm"] == 6800.0

    def test_snapshot_includes_tyre_odometers(self):
        state = make_state()
        snap = state.get_snapshot()
        odometers = snap["player"]["tyre_odometers"]
        assert odometers["LF"] == 1500.0
        assert odometers["RR"] == 1500.0

    def test_track_wetness_labels(self):
        """iRacing TrackWetness is 0-7 enum, not 0-3 scale.
        0=unknown, 1=dry, 2=mostly_dry, 3=very_lightly_wet,
        4=lightly_wet, 5=moderately_wet, 6=very_wet, 7=extremely_wet.
        """
        # Dry track (value 1) should say "Dry", not "Damp"
        assert TRACK_WETNESS_LABELS[1] == "Dry"
        # Unknown (value 0) should be None — don't report if unknown
        assert TRACK_WETNESS_LABELS[0] is None
        # Wet levels
        assert TRACK_WETNESS_LABELS[2] == "Mostly Dry"
        assert TRACK_WETNESS_LABELS[3] == "Very Lightly Wet"
        assert TRACK_WETNESS_LABELS[7] == "Extremely Wet"

    def test_dry_track_not_reported_as_damp(self):
        """A dry track (TrackWetness=1) must appear as 'Dry', never 'Damp'."""
        state = make_state()
        # Set wetness to 1 (dry in iRacing's enum)
        state.session.track_wetness = 1.0
        config = {"prompt": {"context_depth": "full", "system": "test"}}
        builder = ContextBuilder(config)
        content = builder.build_prompt(state.get_snapshot())[1]["content"]
        # Must say "Dry", must NOT say "Damp"
        assert "Dry" in content
        assert "Damp" not in content

    def test_unknown_wetness_not_reported(self):
        """TrackWetness=0 (unknown) should not appear in context at all."""
        state = make_state()
        state.session.track_wetness = 0.0
        config = {"prompt": {"context_depth": "full", "system": "test"}}
        builder = ContextBuilder(config)
        content = builder.build_prompt(state.get_snapshot())[1]["content"]
        # Should not mention track wetness condition at all
        assert "Dry" not in content
        assert "Damp" not in content

    def test_race_ending_soon_suppressed_at_lap_zero(self):
        """RACE ENDING SOON must NOT fire at lap 0 (formation/pre-race).

        At lap 0, iRacing reports near-zero time remaining during the pace lap,
        which makes lap estimates unreliable. The guard prevents the LLM from
        getting "do not pit, stay out and finish" at the start of a 30-min race.
        """
        # Create a time-based race state at lap 0 with very few estimated laps
        config = {"prompt": {"context_depth": "full", "system": "test"}}
        state = RaceState(config)
        tel = sample_telemetry(lap=0, fuel=72.0)
        # Make it a time-based race by setting laps_remain to the sentinel value
        tel["SessionLapsRemain"] = 32767  # sentinel for time-based race
        tel["SessionTimeRemain"] = 114.0  # ~1.9 min left (formation lap)
        tel["RaceLaps"] = 0
        tel["Lap"] = 0
        state.update(tel, sample_session_info(), {0: "Test Driver", 1: "Other Driver"})
        # Force estimated_total_laps to a small number (simulates unreliable estimate)
        state._estimated_total_laps = 2
        snap = state.get_snapshot()
        builder = ContextBuilder(config)
        content = builder.build_prompt(snap)[1]["content"]
        # RACE ENDING SOON must NOT appear at lap 0
        assert "race ending soon" not in content
        assert "do not pit" not in content

    def test_race_ending_soon_fires_late_race(self):
        """RACE ENDING SOON SHOULD fire at lap 10+ with ≤2 laps remaining."""
        config = {"prompt": {"context_depth": "full", "system": "test"}}
        state = RaceState(config)
        tel = sample_telemetry(lap=10, fuel=5.0)
        # Time-based race with very few estimated laps remaining
        tel["SessionLapsRemain"] = 32767  # sentinel for time-based race
        tel["SessionTimeRemain"] = 180.0  # 3 min left
        tel["RaceLaps"] = 10
        tel["Lap"] = 10
        state.update(tel, sample_session_info(), {0: "Test Driver", 1: "Other Driver"})
        # Force estimated_total_laps = 12 → race_laps_remain = 2
        state._estimated_total_laps = 12
        snap = state.get_snapshot()
        builder = ContextBuilder(config)
        content = builder.build_prompt(snap)[1]["content"]
        # race ending soon SHOULD appear at lap 10 with only 2 laps left
        assert "race ending soon" in content
        assert "Wet" not in content


# --- Capture/Replay tests ---


class TestTelemetryReplay:
    def test_load_and_replay(self):
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshots
            for i in range(3):
                filepath = f"{tmpdir}/snapshot_{i + 1:04d}.json"
                with open(filepath, "w") as f:
                    json.dump(
                        {
                            "timestamp": f"2024-01-01T00:00:{i:02d}",
                            "snapshot_num": i + 1,
                            "session_info": {},
                            "telemetry": {"Speed": 85.0 + i, "Lap": i + 1},
                            "driver_names": {"0": "Test"},
                        },
                        f,
                    )

            replay = TelemetryReplay(tmpdir, loop=False)
            count = replay.load()
            assert count == 3

            # Read all snapshots
            snapshots = []
            while replay.has_more():
                snap = replay.next_snapshot()
                if snap is None:
                    break
                snapshots.append(snap)

            assert len(snapshots) == 3
            assert snapshots[0]["telemetry"]["Lap"] == 1
            assert snapshots[2]["telemetry"]["Lap"] == 3

    def test_replay_loop(self):
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                filepath = f"{tmpdir}/snapshot_{i + 1:04d}.json"
                with open(filepath, "w") as f:
                    json.dump(
                        {
                            "timestamp": f"2024-01-01T00:00:{i:02d}",
                            "snapshot_num": i + 1,
                            "session_info": {},
                            "telemetry": {"Lap": i + 1},
                            "driver_names": {},
                        },
                        f,
                    )

            replay = TelemetryReplay(tmpdir, loop=True)
            replay.load()
            # Should loop: 1, 2, 1, 2...
            first = replay.next_snapshot()
            assert first["telemetry"]["Lap"] == 1
            second = replay.next_snapshot()
            assert second["telemetry"]["Lap"] == 2
            third = replay.next_snapshot()  # Loops back
            assert third["telemetry"]["Lap"] == 1


# --- DriverState tests ---


class TestDriverState:
    def test_driver_state_defaults(self):
        ds = DriverState()
        assert ds.car_idx == 0
        assert ds.driver_name == ""
        assert ds.fuel_level == 0.0  # Default for non-player car
        assert ds.position == 0
        assert ds.lap_history == []

    def test_driver_state_with_data(self):
        ds = DriverState(
            car_idx=3,
            driver_name="Dan Golden",
            position=4,
            last_lap_time=93.5,
            best_lap_time=92.3,
            p2p_remaining=2,
            tire_compound=1,
        )
        assert ds.driver_name == "Dan Golden"
        assert ds.last_lap_time == 93.5
        assert ds.p2p_remaining == 2

    def test_player_vs_nearby_car(self):
        """Player car has fuel/tyre detail, nearby cars don't."""
        player = DriverState(
            car_idx=0,
            fuel_level=72.0,
            fuel_pct=0.65,
            tyres={"LF": TyreState(temp_center=97.0)},
        )
        nearby = DriverState(
            car_idx=3,
            last_lap_time=93.5,
            # fuel and tyres left at defaults (0.0, empty dict)
        )
        assert player.fuel_level == 72.0
        assert nearby.fuel_level == 0.0  # Not available for other cars
        assert len(player.tyres) == 1
        assert len(nearby.tyres) == 0


# --- SectorTimeTracker tests ---


class TestSectorTimeTracker:
    """Tests for sector boundary crossing detection and fastest time tracking."""

    def test_set_boundaries(self):
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])
        assert tracker.boundaries is not None
        assert tracker.boundaries.num_sectors == 3
        assert tracker.boundaries.sector_boundaries == [0.0, 0.3333, 0.6667]

    def test_set_boundaries_too_few(self):
        """Setting boundaries with < 2 entries is ignored."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0])
        assert tracker.boundaries is None

    def test_get_sector_for_pct(self):
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])
        # Sector 1: 0.0 <= pct < 0.3333
        assert tracker._get_sector_for_pct(0.0) == 1
        assert tracker._get_sector_for_pct(0.1) == 1
        assert tracker._get_sector_for_pct(0.3332) == 1
        # Sector 2: 0.3333 <= pct < 0.6667
        assert tracker._get_sector_for_pct(0.3333) == 2
        assert tracker._get_sector_for_pct(0.5) == 2
        assert tracker._get_sector_for_pct(0.6666) == 2
        # Sector 3: 0.6667 <= pct < 1.0
        assert tracker._get_sector_for_pct(0.6667) == 3
        assert tracker._get_sector_for_pct(0.9) == 3

    def test_first_tick_baseline_only(self):
        """First tick records position but doesn't compute sector times."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])
        tracker.update(
            lap_dist_pcts=[0.1],
            session_time=10.0,
            track_surfaces=[3],
            on_pit_roads=[0],
            positions=[1],
        )
        # No sector times yet — just baselined the car position
        assert tracker.get_fastest_sectors(0) == {}

    def test_boundary_crossing_detects_sector_time(self):
        """Car crosses a sector boundary → sector time is computed."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Tick 1: Car at 0.1 (sector 1), session time 10.0
        tracker.update(
            lap_dist_pcts=[0.1],
            session_time=10.0,
            track_surfaces=[3],
            on_pit_roads=[0],
            positions=[1],
        )
        # Tick 2: Car at 0.4 (crossed boundary at 0.3333, now in sector 2)
        tracker.update(
            lap_dist_pcts=[0.4],
            session_time=38.5,
            track_surfaces=[3],
            on_pit_roads=[0],
            positions=[1],
        )

        # Sector 1 time should be ~28.5s (38.5 - 10.0)
        sectors = tracker.get_fastest_sectors(0)
        assert 1 in sectors
        assert abs(sectors[1] - 28.5) < 0.5

    def test_multiple_sectors(self):
        """Car completes all three sectors in a lap."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Tick 1: baseline at 0.1
        tracker.update([0.1], 10.0, [2], [0], [1])
        # Tick 2: crosses S1 boundary at 0.3333
        tracker.update([0.4], 38.5, [2], [0], [1])
        # Tick 3: crosses S2 boundary at 0.6667
        tracker.update([0.8], 70.0, [2], [0], [1])

        sectors = tracker.get_fastest_sectors(0)
        assert 1 in sectors
        assert abs(sectors[1] - 28.5) < 0.5
        assert 2 in sectors
        assert abs(sectors[2] - 31.5) < 0.5

    def test_lap_wraparound(self):
        """Car wraps from end of lap to start of new lap (0.98 → 0.05)."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Build up some state
        tracker.update([0.8], 10.0, [2], [0], [1])
        # Car approaches end of lap
        tracker.update([0.95], 15.0, [2], [0], [1])
        # Car wraps around (crosses start/finish = boundary 0.0)
        tracker.update([0.05], 22.0, [2], [0], [1])

        # Sector 3 time should be computed (~12s from 10.0 to 22.0 minus
        # time spent before entering sector 3)
        sectors = tracker.get_fastest_sectors(0)
        # The exact value depends on which sector the car was in,
        # but sector 3 should have been completed
        assert 3 in sectors
        assert sectors[3] > 0

    def test_car_in_pits_ignored(self):
        """Car on pit road doesn't trigger sector crossings."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Baseline
        tracker.update([0.1], 10.0, [2], [0], [1])
        # Car enters pits
        tracker.update([0.4], 38.5, [2], [1], [1])
        sectors = tracker.get_fastest_sectors(0)
        # No sector time — was in pits when boundary was crossed
        assert 1 not in sectors

    def test_car_off_track_ignored(self):
        """Car with LapDistPct == -1 (off track) is skipped."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Baseline
        tracker.update([0.1], 10.0, [2], [0], [1])
        # Car goes off track
        tracker.update([-1.0], 38.5, [0], [0], [1])
        # Car comes back on track at different position
        tracker.update([0.5], 50.0, [2], [0], [1])
        # No sector time from the off-track period
        sectors = tracker.get_fastest_sectors(0)
        assert 1 not in sectors

    def test_fastest_sector_updates(self):
        """Completing a sector faster than previous best updates it."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Lap 1: S1 = 30s
        tracker.update([0.1], 10.0, [2], [0], [1])
        tracker.update([0.4], 40.0, [2], [0], [1])

        # Continue lap 1: complete S2, S3 and wrap to start new lap
        tracker.update([0.8], 70.0, [2], [0], [1])
        tracker.update([0.95], 85.0, [2], [0], [1])
        # Cross start/finish into new lap
        tracker.update([0.05], 100.0, [2], [0], [1])

        # Lap 2: S1 = 28s (faster than the ~30s from lap 1)
        tracker.update([0.4], 128.0, [2], [0], [1])

        sectors = tracker.get_fastest_sectors(0)
        assert 1 in sectors
        # The fastest S1 should be ~28s (128.0 - 100.0)
        assert abs(sectors[1] - 28.0) < 1.0  # Faster time wins

    def test_partial_sector_rejected(self):
        """A sector time less than half the existing best is rejected as partial."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        # Set up: enter S3 at 70s, complete it at 117s → S3 = 47s (legitimate)
        tracker.update([0.1], 10.0, [2], [0], [1])  # S1 enter
        tracker.update([0.4], 40.0, [2], [0], [1])  # S2 enter, S1 = 30s
        tracker.update([0.7], 70.0, [2], [0], [1])  # S3 enter, S2 = 30s
        tracker.update([0.05], 117.0, [2], [0], [1])  # S1 enter, S3 = 47s
        sectors = tracker.get_fastest_sectors(0)
        assert 3 in sectors
        assert abs(sectors[3] - 47.0) < 1.0

        # Now try to update S3 with a partial time of 20s (< 47 * 0.5 = 23.5)
        # This should be REJECTED — it's from an incomplete sector
        tracker.update([0.7], 137.0, [2], [0], [1])  # Re-enter S3
        tracker.update([0.05], 157.0, [2], [0], [1])  # S3 = 20s — should be rejected
        sectors = tracker.get_fastest_sectors(0)
        # S3 should still be ~47s, not 20s
        assert abs(sectors[3] - 47.0) < 1.0

        # But a legitimate improvement (e.g., 45s) should be accepted
        tracker.update([0.7], 177.0, [2], [0], [1])  # Re-enter S3
        tracker.update([0.05], 222.0, [2], [0], [1])  # S3 = 45s — should be accepted
        sectors = tracker.get_fastest_sectors(0)
        assert abs(sectors[3] - 45.0) < 1.0

    def test_invalid_position_skipped(self):
        """Car with position <= 0 is skipped (disconnected)."""
        tracker = SectorTimeTracker()
        tracker.set_boundaries([0.0, 0.3333, 0.6667])

        tracker.update([0.1], 10.0, [2], [0], [0])  # pos=0 → invalid
        assert 0 not in tracker._car_states

    def test_get_fastest_sectors_empty(self):
        """No data for unknown car returns empty dict."""
        tracker = SectorTimeTracker()
        assert tracker.get_fastest_sectors(99) == {}

    def test_snapshot_includes_sector_times(self):
        """get_snapshot() includes fastest_sector_times for player and nearby cars."""
        state = make_state()
        # After update, sector_tracker should have boundaries from SplitTimeInfo
        assert state.sector_tracker.boundaries is not None
        snap = state.get_snapshot()
        assert "fastest_sector_times" in snap["player"]
        assert isinstance(snap["player"]["fastest_sector_times"], dict)
        for car in snap["nearby_cars"]:
            assert "fastest_sector_times" in car


# --- LLMClient retry tests ---


class TestLLMClientRetry:
    """Tests for LLMClient transient-error retry logic."""

    def _make_client(self, retries=2, backoff=0.0, thinking=None):
        from llm_client import LLMClient

        config = {
            "llm": {
                "base_url": "http://localhost:1",  # nothing listening
                "api_key": "test",
                "model": "test",
                "retries": retries,
                "retry_backoff": backoff,
                "timeout": 0.1,
                "thinking": thinking,
            }
        }
        return LLMClient(config)

    @staticmethod
    def _make_response(text):
        class Msg:
            content = text

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]
            usage = None

        return Resp()

    @staticmethod
    def _make_api_error(exc_cls, message):
        """Build an openai SDK error — needs a response with .request set."""

        class Request:
            method = "POST"
            url = "http://localhost:1/v1/chat/completions"

        class Response:
            request = Request()
            status_code = 429
            headers = {}

        return exc_cls(message, response=Response(), body=None)

    def test_success_first_try(self):
        client = self._make_client()
        calls = []
        client.client.chat.completions.create = lambda *a, **kw: (
            calls.append(1) or self._make_response("pit now")
        )
        assert client.ask([{"role": "user", "content": "q"}]) == "pit now"
        assert len(calls) == 1

    def test_retry_then_success(self):
        """A transient error followed by success returns the good response."""
        import openai

        client = self._make_client(retries=2, backoff=0.0)
        attempts = []

        def flaky(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise self._make_api_error(openai.RateLimitError, "rate limit exceeded")
            return self._make_response("stay out")

        client.client.chat.completions.create = flaky
        assert client.ask([{"role": "user", "content": "q"}]) == "stay out"
        assert len(attempts) == 2

    def test_transient_exhausted_raises_llmerror(self):
        """Persistent transient errors exhaust retries and raise LLMError."""
        import openai

        client = self._make_client(retries=1, backoff=0.0)
        attempts = []

        def always_rate_limited(*args, **kwargs):
            attempts.append(1)
            raise self._make_api_error(openai.RateLimitError, "rate limit exceeded")

        client.client.chat.completions.create = always_rate_limited
        from llm_client import LLMError

        try:
            client.ask([{"role": "user", "content": "q"}])
            assert False, "expected LLMError"
        except LLMError:
            pass
        assert len(attempts) == 2  # initial + 1 retry

    def test_auth_error_no_retry(self):
        """Auth failures fail fast — no pointless retries."""
        import openai

        client = self._make_client(retries=2, backoff=0.0)
        attempts = []

        def bad_auth(*args, **kwargs):
            attempts.append(1)
            raise self._make_api_error(openai.AuthenticationError, "invalid api key")

        client.client.chat.completions.create = bad_auth
        from llm_client import LLMError

        try:
            client.ask([{"role": "user", "content": "q"}])
            assert False, "expected LLMError"
        except LLMError:
            pass
        assert len(attempts) == 1  # no retry on permanent errors

    def test_timeout_message(self):
        """Timeout errors surface the friendly 'I'm busy' message."""
        client = self._make_client(retries=0)

        def slow(*args, **kwargs):
            raise Exception("Request timed out after 100ms")

        client.client.chat.completions.create = slow
        from llm_client import LLMError

        try:
            client.ask([{"role": "user", "content": "q"}])
            assert False, "expected LLMError"
        except LLMError as e:
            assert "busy" in str(e).lower()

    def test_thinking_sent_when_set(self):
        """thinking config passes both known spellings via extra_body."""
        client = self._make_client(thinking="high")
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return self._make_response("ok")

        client.client.chat.completions.create = capture
        client.ask([{"role": "user", "content": "q"}])
        extra = captured.get("extra_body", {})
        assert extra.get("think") == "high"
        assert extra.get("reasoning_effort") == "high"

    def test_thinking_false_maps_to_low_effort(self):
        """thinking: false sends think=false; OpenAI's reasoning_effort (strings
        only) gets "low" as the closest equivalent."""
        client = self._make_client(thinking=False)
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return self._make_response("ok")

        client.client.chat.completions.create = capture
        client.ask([{"role": "user", "content": "q"}])
        extra = captured.get("extra_body", {})
        assert extra.get("think") is False
        assert extra.get("reasoning_effort") == "low"

    def test_thinking_string_aliases(self):
        """String spellings normalise: "False"/"off"/"none" → False."""
        for raw in ("False", "OFF", "none"):
            client = self._make_client(thinking=raw)
            assert client.thinking is False, f"{raw!r} should map to False"
        client = self._make_client(thinking="HIGH")
        assert client.thinking == "high"
        client = self._make_client(thinking=True)
        assert client.thinking is True

    def test_thinking_omitted_when_unset(self):
        """No thinking keys sent when thinking is unset/empty."""
        for raw in (None, ""):
            client = self._make_client(thinking=raw)
            captured = {}

            def capture(**kwargs):
                captured.update(kwargs)
                return self._make_response("ok")

            client.client.chat.completions.create = capture
            client.ask([{"role": "user", "content": "q"}])
            assert "extra_body" not in captured


# --- Eval quality-check tests ---


class TestEvalQualityChecks:
    """Tests for tests/eval_llm_responses.py check_response_quality."""

    @staticmethod
    def _check(question, response, context):
        from tests.eval_llm_responses import check_response_quality

        return check_response_quality(question, response, context)

    def test_empty_response_is_infrastructure_failure(self):
        from tests.eval_llm_responses import is_infrastructure_failure

        assert is_infrastructure_failure("")
        assert is_infrastructure_failure("(empty)")
        assert is_infrastructure_failure("[ERROR: I'm busy, try again in a minute.]")
        assert not is_infrastructure_failure("Pit now for fuel.")

    def test_infrastructure_failure_not_scored(self):
        """ERROR-wrapped and empty responses are excluded from model scoring."""
        issues = self._check(
            "When should I pit?", "[ERROR: I'm busy, try again in a minute.]", ""
        )
        assert issues == []

    def test_margin_vs_range_confusion_detected(self):
        """The 'only X laps margin' trap: margin is not total fuel remaining."""
        context = (
            "Fuel: 4.1 litres/22 litre tank. Burn approx 1.31 litres/lap, "
            "range approx 3.2 laps.\n"
            "!! Fuel tight: only 0.2 laps margin. conserve fuel."
        )
        issues = self._check(
            "Any warnings I should know about?",
            "Fuel is critical – you have only ~0.2 laps of fuel left, so you must conserve.",
            context,
        )
        assert any("Margin-vs-range confusion" in i for i in issues)

    def test_margin_vs_range_correct_answer_passes(self):
        context = (
            "Fuel: 4.1 litres/22 litre tank. Burn approx 1.31 litres/lap, "
            "range approx 3.2 laps.\n"
            "!! Fuel tight: only 0.2 laps margin. conserve fuel."
        )
        issues = self._check(
            "Any warnings I should know about?",
            "Fuel is critical—only ~0.2 laps margin, so conserve and stay out.",
            context,
        )
        assert not any("Margin-vs-range" in i for i in issues)

    def test_gap_vs_pace_confusion_detected(self):
        """Calling the gap to the car behind a 'pace difference' is wrong."""
        context = (
            "Nearby (+ ahead, − behind):\n"
            "  P10 (Nick Leep): -5.124s behind, last lap 2:27.669\n"
        )
        issues = self._check(
            "Am I safe from the car behind?",
            "You're safe – the car behind is 5.1s slower and can't close the gap.",
            context,
        )
        assert any("Gap-vs-pace confusion" in i for i in issues)

    def test_gap_vs_pace_correct_answer_passes(self):
        context = (
            "Nearby (+ ahead, − behind):\n"
            "  P10 (Nick Leep): -5.124s behind, last lap 2:27.669\n"
        )
        issues = self._check(
            "Am I safe from the car behind?",
            "You're safe – P10 is 5.1s behind and lapping much slower.",
            context,
        )
        assert not any("Gap-vs-pace" in i for i in issues)

    def test_direction_inversion_detected(self):
        """Describing a car ahead as behind (or vice versa) is flagged."""
        context = (
            "Nearby (+ ahead, − behind):\n"
            "  P3 (Nik McCarter): +2.174s ahead, last lap 2:22.708\n"
            "  P5 (Kim Berry): -1.023s behind, last lap 2:25.083\n"
        )
        issues = self._check(
            "Should I use push-to-pass?",
            "Yes — use push-to-pass now to defend P3 and attack P5.",
            context,
        )
        # P3 is ahead (+2.174s) — you can't "defend" against it
        assert any("Direction inversion" in i for i in issues)

    def test_direction_inversion_correct_passes(self):
        context = (
            "Nearby (+ ahead, − behind):\n"
            "  P3 (Nik McCarter): +2.174s ahead, last lap 2:22.708\n"
            "  P5 (Kim Berry): -1.023s behind, last lap 2:25.083\n"
        )
        issues = self._check(
            "Should I use push-to-pass?",
            "No — save fuel; you're 2.174s behind P3 and P5 is 1.023s behind you.",
            context,
        )
        assert not any("Direction inversion" in i for i in issues)

    def test_refusal_phrase_flagged(self):
        issues = self._check("How are we looking?", "I'm busy, try again later.", "")
        assert any("Refusal" in i for i in issues)

    def test_clean_response_no_issues(self):
        context = (
            "Fuel: 5.2 litres/22 litre tank. Burn approx 1.31 litres/lap, "
            "range approx 4.0 laps."
        )
        issues = self._check(
            "How many laps of fuel left?", "Approximately 4 laps of fuel left.", context
        )
        assert issues == []


class TestChatMacroManager:
    """Tests for ChatMacroManager — config parsing, app.ini validation,
    [CMD:] tag extraction, and macro firing."""

    CONFIG = {
        "chat_macros": {
            "actions": {
                "clear_black_flag": {
                    "macro": 15,
                    "command": "!clearall",
                    "description": "Clears your black flags",
                },
                "pitting_in": {
                    "macro": 1,
                    "command": "Pitting In",
                    "description": "Tells the field you're pitting",
                },
            }
        }
    }

    def _manager(self, config=None):
        from chat_macros import ChatMacroManager

        return ChatMacroManager(config if config is not None else self.CONFIG)

    def test_parses_configured_actions(self):
        m = self._manager()
        assert m.enabled
        assert "clear_black_flag" in m.actions
        assert m.actions["clear_black_flag"]["macro"] == 15
        assert m.actions["pitting_in"]["macro"] == 1

    def test_empty_config_disables(self):
        m = self._manager({})
        assert not m.enabled
        assert m.actions == {}

    def test_invalid_macro_number_dropped(self):
        config = {
            "chat_macros": {
                "actions": {
                    "bad_zero": {"macro": 0, "command": "x"},
                    "bad_high": {"macro": 16, "command": "x"},
                    "bad_str": {"macro": "five", "command": "x"},
                    "good": {"macro": 3, "command": "x"},
                }
            }
        }
        m = self._manager(config)
        assert list(m.actions) == ["good"]

    def test_parse_response_extracts_and_strips_tags(self):
        m = self._manager()
        clean, actions = m.parse_response(
            "Clearing your flags now. [CMD:clear_black_flag]"
        )
        assert actions == ["clear_black_flag"]
        assert clean == "Clearing your flags now."
        assert "[CMD" not in clean

    def test_parse_response_multiple_tags(self):
        m = self._manager()
        clean, actions = m.parse_response(
            "[CMD:clear_black_flag] [CMD:pitting_in] Boxing this lap."
        )
        assert actions == ["clear_black_flag", "pitting_in"]
        assert clean == "Boxing this lap."

    def test_parse_response_unknown_action_dropped(self):
        m = self._manager()
        clean, actions = m.parse_response("Doing it. [CMD:make_tea]")
        assert actions == []
        assert clean == "Doing it."

    def test_parse_response_no_tags_unchanged(self):
        m = self._manager()
        clean, actions = m.parse_response("Stay out for two more laps.")
        assert actions == []
        assert clean == "Stay out for two more laps."

    def test_parse_response_empty_text(self):
        m = self._manager()
        clean, actions = m.parse_response("")
        assert clean == ""
        assert actions == []

    def test_fire_log_only_without_client(self):
        # iracing_client=None (replay/tests) — logs intent, returns True
        m = self._manager()
        assert m.fire("clear_black_flag") is True
        assert m.fire("nonexistent") is False

    def test_fire_sends_macro_via_client(self):
        m = self._manager()

        class FakeClient:
            def __init__(self):
                self.sent = []

            def send_chat_macro(self, macro_num):
                self.sent.append(macro_num)

        client = FakeClient()
        assert m.fire("clear_black_flag", iracing_client=client) is True
        assert client.sent == [15]

    def test_available_actions_prompt_lists_all(self):
        m = self._manager()
        prompt = m.available_actions_prompt()
        assert "clear_black_flag" in prompt
        assert "pitting_in" in prompt
        assert "[CMD:clear_black_flag]" in prompt  # example tag included
        assert "Never invent command names" in prompt

    def test_available_actions_prompt_empty_when_disabled(self):
        m = self._manager({})
        assert m.available_actions_prompt() == ""

    def test_validate_against_app_ini_matching(self, tmp_path):
        app_ini = tmp_path / "app.ini"
        app_ini.write_text(
            "[Autochat Messages]\nAutoChatStr1=Pitting In$\nAutoChatStr15=!clearall$\n",
            encoding="utf-8",
        )
        config = {
            "chat_macros": {
                "app_ini_path": str(app_ini),
                "actions": self.CONFIG["chat_macros"]["actions"],
            }
        }
        m = self._manager(config)
        assert m.validate_against_app_ini(config) == []

    def test_validate_against_app_ini_mismatch(self, tmp_path):
        app_ini = tmp_path / "app.ini"
        app_ini.write_text(
            "[Autochat Messages]\nAutoChatStr15=Wrong command$\n",
            encoding="utf-8",
        )
        config = {
            "chat_macros": {
                "app_ini_path": str(app_ini),
                "actions": self.CONFIG["chat_macros"]["actions"],
            }
        }
        m = self._manager(config)
        warnings = m.validate_against_app_ini(config)
        assert any("Wrong command" in w for w in warnings)

    def test_validate_missing_app_ini(self, tmp_path):
        config = {
            "chat_macros": {
                "app_ini_path": str(tmp_path / "nonexistent.ini"),
                "actions": self.CONFIG["chat_macros"]["actions"],
            }
        }
        m = self._manager(config)
        warnings = m.validate_against_app_ini(config)
        assert any("not found" in w for w in warnings)

    def test_validate_handles_inline_comments(self, tmp_path):
        # app.ini has "; comment" suffixes — parser must strip them
        app_ini = tmp_path / "app.ini"
        app_ini.write_text(
            "[Autochat Messages]\n"
            "AutoChatStr1=Pitting In$          ; autochat message\n"
            "AutoChatStr15=!clearall$          ; autochat message\n",
            encoding="utf-8",
        )
        config = {
            "chat_macros": {
                "app_ini_path": str(app_ini),
                "actions": self.CONFIG["chat_macros"]["actions"],
            }
        }
        m = self._manager(config)
        assert m.validate_against_app_ini(config) == []


class TestChatMacroPromptInjection:
    """Tests for chat macros block injection into the system prompt."""

    def _builder_with_macros(self):
        from chat_macros import ChatMacroManager

        config = {
            "prompt": {"system": "Race engineer. Be brief."},
            "chat_macros": {
                "actions": {
                    "clear_black_flag": {
                        "macro": 15,
                        "command": "!clearall",
                        "description": "Clears your black flags",
                    }
                }
            },
        }
        builder = ContextBuilder(config)
        manager = ChatMacroManager(config)
        builder.set_chat_macros(manager)
        return builder

    def _state(self):
        return make_state().get_snapshot()

    def test_system_prompt_contains_actions_block(self):
        builder = self._builder_with_macros()
        messages = builder.build_prompt(self._state(), question="clear my flags")
        system = messages[0]["content"]
        assert "clear_black_flag" in system
        assert "[CMD:clear_black_flag]" in system

    def test_system_prompt_without_macros_unchanged(self):
        builder = ContextBuilder({"prompt": {"system": "Race engineer. Be brief."}})
        messages = builder.build_prompt(self._state(), question="hello")
        assert messages[0]["content"] == "Race engineer. Be brief."
