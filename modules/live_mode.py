"""
Live mode — connect to iRacing, poll telemetry, and service wheel-button
LLM queries / voice push-to-talk.

Extracted from main.py: voice client setup now lives in voice_setup.py, the
push-to-talk closures in push_to_talk.py, and the LLM query flow in
engineer_query.py. This module keeps only the poll loop, trigger wiring, and
the spotter tick.
"""

import logging
import threading
import time

from modules.chat_macros import ChatMacroManager
from modules.context_builder import ContextBuilder
from modules.engineer_query import _state_lock, handle_button_press
from modules.iracing_client import IRacingClient
from modules.llm_client import LLMClient
from modules.push_to_talk import PushToTalkController
from modules.race_state import RaceState
from modules.spotter import Spotter
from modules.voice_setup import preload_voice_models, setup_stt, setup_tts
from modules.wheel_button import WheelButtonListener

logger = logging.getLogger("iraceengineer")


def _tick_spotter(state: RaceState, spotter) -> None:
    """Feed the current race state to the spotter (car proximity + alerts).

    Selects the closest car directly behind the player for the closing-approach
    alert, then calls spotter.update() with all the values it needs.
    """
    car_lr = state.player.car_left_right
    on_track = state.player.is_on_track
    track_surface = state.player.player_track_surface
    fuel_laps = state.fuel_laps_remaining
    flags = state.session.flags
    session_state = state.session.session_state
    incidents = state.player.incidents
    on_pit_road = state.player.on_pit_road

    # Find car directly behind player for closing-approach alert
    car_behind_gap = 0.0
    car_behind_lap_time = -1.0
    car_behind_on_pit_road = False
    for car in state.nearby_cars:
        if car.gap_seconds < 0:  # negative = behind player
            # Closest behind = least negative gap
            if car_behind_gap == 0.0 or car.gap_seconds > car_behind_gap:
                car_behind_gap = car.gap_seconds
                car_behind_lap_time = car.last_lap_time
                car_behind_on_pit_road = getattr(car, "on_pit_road", False)

    spotter.update(
        car_lr,
        is_on_track=on_track,
        track_surface=track_surface,
        fuel_laps_remaining=fuel_laps,
        session_flags=flags,
        session_state=session_state,
        incidents=incidents,
        on_pit_road=on_pit_road,
        car_behind_gap=car_behind_gap,
        car_behind_lap_time=car_behind_lap_time,
        car_behind_gap_metres=state.player.car_dist_behind,
        player_last_lap_time=state.player.last_lap_time,
        player_best_lap_time=state.player.best_lap_time,
        car_behind_on_pit_road=car_behind_on_pit_road,
        lap_completed=state.player.lap_completed,
    )


def _setup_spotter(config: dict) -> Spotter | None:
    """Gracefully construct the spotter — None if disabled or unavailable."""
    spotter_config = config.get("spotter", {})
    if not spotter_config.get("enabled", False):
        return None
    try:
        spotter = Spotter(config)
        if not spotter.enabled:
            return None
        logger.info("Spotter enabled — car proximity calls active")
        return spotter
    except Exception as e:
        logger.warning(f"Spotter setup failed: {e}")
        return None


def run_live_mode(config: dict, tick_rate_hz: int = 30, question: str = ""):
    """Main loop — connect to iRacing and poll telemetry."""
    iracing = IRacingClient()
    state = RaceState(config)
    context_builder = ContextBuilder(config)
    llm = LLMClient(config)

    # Set up voice clients (graceful if not installed)
    tts = setup_tts(config)
    stt = setup_stt(config)

    # Set up spotter (real-time audio calls for car proximity)
    spotter = _setup_spotter(config)

    # Register CUDA DLLs and pre-load voice models to avoid cold-start latency
    preload_voice_models(stt, tts)

    # Set up wheel button triggers
    voice_config = config.get("voice", {})
    voice_trigger_config = voice_config.get("trigger", {})

    # Connect to iRacing
    print("\n🔌 Connecting to iRacing...")
    if not iracing.startup():
        print(
            "❌ iRacing not found. Make sure iRacing is running with an active session."
        )
        print("   Retrying every 5 seconds... Press Ctrl+C to exit.")

        while not iracing.startup():
            time.sleep(5)
            if iracing.is_connected:
                break

    print("✅ Connected to iRacing!\n")

    # Detect teammates
    team_indices = iracing.detect_team_indices(config)
    driver_aliases = getattr(iracing, "driver_aliases", {})
    state.set_team_indices(team_indices, driver_aliases)
    driver_names = {d.car_idx: d.driver_name for d in iracing.drivers}
    state.set_driver_names(driver_names)
    print(
        f"👥 Team detected: {len(team_indices)} cars — {', '.join(driver_names.get(i, f'Car #{i}') for i in team_indices)}\n"
    )

    # Chat macro actions (e.g. clear black flag) — validate against app.ini
    # so misconfigured macro slots are caught before they matter mid-race
    chat_macros = ChatMacroManager(config)
    if chat_macros.enabled:
        chat_macros.validate_against_app_ini(config)
        context_builder.set_chat_macros(chat_macros)
        print(f"⚙️  Chat macro actions: {', '.join(chat_macros.actions)}\n")

    # Register triggers (wheel button for voice PTT / LLM query)
    voice_listener = None

    # If a question was given on the command line, ask it once state is live
    if question:

        def _ask_cli_question():
            # Wait a moment for telemetry to populate state
            time.sleep(1.0)
            handle_button_press(
                state,
                context_builder,
                llm,
                question=question,
                tts=tts,
                chat_macros=chat_macros,
                iracing=iracing,
            )

        threading.Thread(target=_ask_cli_question, daemon=True).start()
        print(f'   Will ask: "{question}"\n')

    # Voice PTT trigger — wheel button hold-to-talk
    if stt is not None:
        voice_device = voice_trigger_config.get("device_index")
        voice_button = voice_trigger_config.get("button_index")
        if voice_device is not None and voice_button is not None:

            def _on_voice_question(text, timing_steps):
                handle_button_press(
                    state,
                    context_builder,
                    llm,
                    question=text,
                    tts=tts,
                    timing_steps=timing_steps,
                    chat_macros=chat_macros,
                    iracing=iracing,
                )

            ptt = PushToTalkController(
                stt,
                on_question=_on_voice_question,
                max_record_seconds=voice_trigger_config.get("max_record_seconds", 15),
            )
            voice_listener = WheelButtonListener(
                device_index=voice_device,
                button_index=voice_button,
                on_press=ptt.on_press,
                on_release=ptt.on_release,
            )
            voice_listener.start()
            print(
                f"   Voice PTT: hold wheel button {voice_button} (device {voice_device}) "
                f"to speak your question."
            )
        else:
            logger.warning(
                "Voice enabled but no wheel button trigger configured — "
                "set voice.trigger.device_index and voice.trigger.button_index."
            )
            print(
                "   ⚠️  Voice PTT not configured. Edit config.yaml voice.trigger section."
            )

    # Non-voice fallback: wheel button press = direct LLM query.
    # Only wired when the voice PTT listener isn't already using the button
    # (STT unavailable, or voice disabled via --no-voice).
    if voice_listener is None:
        query_device = voice_trigger_config.get("device_index")
        query_button = voice_trigger_config.get("button_index")
        if query_device is not None and query_button is not None:

            def _on_query_button_down():
                handle_button_press(
                    state,
                    context_builder,
                    llm,
                    question="",
                    tts=tts,
                    chat_macros=chat_macros,
                    iracing=iracing,
                )

            voice_listener = WheelButtonListener(
                device_index=query_device,
                button_index=query_button,
                on_press=_on_query_button_down,
            )
            voice_listener.start()
            print(
                f"   Wheel button {query_button} (device {query_device}) "
                f"press = ask LLM."
            )
        else:
            logger.warning(
                "No wheel button trigger configured — LLM queries only via "
                "--question. Set voice.trigger.device_index and "
                "voice.trigger.button_index."
            )
            print(
                "   ⚠️  No query trigger configured. Use --question, or set "
                "voice.trigger.device_index and voice.trigger.button_index."
            )

    print("   Press Ctrl+C to exit.\n")
    tick_interval = 1.0 / tick_rate_hz

    # Cached per-session data: units are static for the whole session, and
    # driver names only change when session info updates. Refresh both when
    # the session info tick changes (reconnects also refresh — see below).
    cached_units: dict = {}
    cached_driver_names: dict = {}
    cached_session_info_tick = None
    _prev_session_num: int | None = None

    try:
        while True:
            if iracing.is_connected:
                # Read telemetry — only the ~100 vars RaceState consumes,
                # not all 327 (each pyirsdk read re-sorts its buffer list)
                telemetry = iracing.get_live_telemetry()
                session_info = iracing.get_session_info()

                # Refresh cached units/driver names when session info changes
                session_info_tick = getattr(iracing, "_session_info_tick", None)
                if session_info_tick != cached_session_info_tick:
                    cached_session_info_tick = session_info_tick
                    try:
                        cached_units = iracing.get_telemetry_units()
                    except Exception:
                        cached_units = {}
                    cached_driver_names = {
                        d.car_idx: d.driver_name for d in iracing.drivers
                    }

                # Update race state (under lock to prevent stale reads from button handler).
                # Returns False when the session just ended — state is frozen
                # at last known values, so skip the spotter too (otherwise it
                # would replay "car still there" forever on frozen data).
                with _state_lock:
                    state_fresh = state.update(
                        telemetry, session_info, cached_driver_names, units=cached_units
                    )

                # Reset spotter when a new session loads (iRacing stays
                # connected across a weekend's sessions — without this,
                # flag/fuel/lap state leaks from practice into the race)
                session_num = state.session.session_num
                if (
                    spotter is not None
                    and _prev_session_num is not None
                    and session_num != _prev_session_num
                ):
                    logger.info(
                        f"Session change detected ({_prev_session_num} → "
                        f"{session_num}) — resetting spotter"
                    )
                    spotter.reset()
                if _prev_session_num is not None and session_num != _prev_session_num:
                    # Advice from practice doesn't apply to the race
                    context_builder.qa_history.clear()
                _prev_session_num = session_num

                # Spotter tick — car proximity audio calls + fuel/flag/pit alerts
                if spotter is not None and state_fresh:
                    _tick_spotter(state, spotter)

            else:
                # iRacing disconnected — try to reconnect
                logger.warning("iRacing disconnected, attempting reconnect...")
                if not iracing.startup():
                    time.sleep(2)
                    continue
                print("✅ Reconnected to iRacing")
                # Re-run session setup: car indices can change between
                # sessions, so team detection and driver names must be
                # re-derived, not just spotter transitions reset
                team_indices = iracing.detect_team_indices(config)
                driver_aliases = getattr(iracing, "driver_aliases", {})
                with _state_lock:
                    state.set_team_indices(team_indices, driver_aliases)
                    state.set_driver_names(
                        {d.car_idx: d.driver_name for d in iracing.drivers}
                    )
                    state.reset()  # drop stale lap history/fuel baselines
                # Invalidate cached per-session data
                cached_session_info_tick = None
                if spotter is not None:
                    spotter.reset()
                print(f"👥 Team re-detected: {len(team_indices)} cars")

            time.sleep(tick_interval)

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if voice_listener is not None:
            voice_listener.stop()
        iracing.shutdown()
