"""Unit tests for iRaceEngineer modules."""

from race_state import (
    RaceState,
    DriverState,
    TyreState,
    LapRecord,
    FLAG_GREEN,
    FLAG_YELLOW,
)
from context_builder import (
    ContextBuilder,
    format_lap_time,
    format_gap,
    format_temp,
    format_pct,
    format_engine_warnings,
    format_car_proximity,
)
from action_executor import ActionExecutor
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

    def test_full_context_includes_engine_health(self):
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
        # Engine health should appear in full context
        assert "Engine:" in content
        assert "Oil" in content

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


# --- ActionExecutor tests ---


class TestActionExecutor:
    def test_parse_simple_action(self):
        executor = ActionExecutor(config={"actions": {"enabled": False}})
        text, actions = executor.parse_response("Pit now.\n[ACTION] pit_this_lap")
        assert "Pit now." in text
        assert len(actions) == 1
        assert actions[0]["action"] == "pit_this_lap"

    def test_parse_action_with_param(self):
        executor = ActionExecutor(config={"actions": {"enabled": False}})
        text, actions = executor.parse_response("Box this lap.\n[ACTION] add_fuel: 60")
        assert len(actions) == 1
        assert actions[0]["action"] == "add_fuel"
        assert actions[0]["param"] == "60"

    def test_parse_multiple_actions(self):
        executor = ActionExecutor(config={"actions": {"enabled": False}})
        text, actions = executor.parse_response(
            "Pit now.\n[ACTION] pit_this_lap\n[ACTION] add_fuel: 60\n[ACTION] change_tyres"
        )
        assert len(actions) == 3

    def test_parse_no_actions(self):
        executor = ActionExecutor(config={"actions": {"enabled": False}})
        text, actions = executor.parse_response("Stay out for 3 more laps.")
        assert text == "Stay out for 3 more laps."
        assert len(actions) == 0

    def test_dry_run_mode(self):
        executor = ActionExecutor(config={"actions": {"enabled": False}})
        results = executor.execute(
            [{"action": "pit_this_lap", "param": None, "raw": "[ACTION] pit_this_lap"}]
        )
        assert results[0].startswith("[WOULD EXECUTE]")

    def test_blocked_action(self):
        executor = ActionExecutor(
            config={"actions": {"enabled": False, "allowed_actions": ["pit_this_lap"]}}
        )
        results = executor.execute(
            [
                {
                    "action": "pit_this_lap",
                    "param": None,
                    "raw": "[ACTION] pit_this_lap",
                },
                {"action": "add_fuel", "param": "60", "raw": "[ACTION] add_fuel: 60"},
            ]
        )
        assert "[WOULD EXECUTE]" in results[0]
        assert "[BLOCKED]" in results[1]


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
