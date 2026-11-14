"""
Replay mode — feed captured telemetry through the pipeline for testing.

If `--question` is provided, process all snapshots then ask once. Otherwise
enter an interactive mode where you can type questions, change context depth,
advance snapshots, or use voice input.

Both query paths (one-shot + interactive) share the ask_engineer() seam from
engineer_query.py — chat macro actions fire log-only here (no live sim).
"""

import logging
import threading
import time

from modules.capture import TelemetryReplay
from modules.chat_macros import ChatMacroManager
from modules.context_builder import ContextBuilder
from modules.engineer_query import _state_lock, ask_engineer
from modules.llm_client import LLMClient
from modules.race_state import RaceState, TyreState
from modules.voice_setup import setup_stt, setup_tts

logger = logging.getLogger("iraceengineer")


def run_replay_mode(
    config: dict,
    replay_dir: str,
    loop: bool = True,
    question: str | None = None,
    replay_speed: float = 0,
):
    """Replay mode — feed captured data through the pipeline.

    If question is provided, process all snapshots then ask once.
    Otherwise, enter interactive mode where you can type questions at any time.

    replay_speed controls how fast snapshots are fed in interactive mode:
        0 = load all instantly (legacy behaviour), then interactive
        N = feed one snapshot every N seconds, state evolves over time
    """
    context_builder = ContextBuilder(config)
    llm = LLMClient(config)
    state = RaceState(config)

    # Chat macro actions — log-only in replay (no live sim to send through)
    chat_macros = ChatMacroManager(config)
    if chat_macros.enabled:
        chat_macros.validate_against_app_ini(config)
        context_builder.set_chat_macros(chat_macros)

    # Set up TTS for replay mode (graceful if not installed)
    tts = setup_tts(config)

    print(f"\n🔄 Loading replay data from {replay_dir}...")
    replay = TelemetryReplay(replay_dir, loop=loop)
    count = replay.load()
    print(f"✅ Loaded {count} snapshots\n")

    # Process snapshots to build state
    # For question mode or instant replay (speed=0), load all upfront.
    # For timed replay (speed>0), load only the first snapshot — the rest
    # are fed one at a time in the interactive loop.
    snapshots_processed = 0
    snapshots_to_preload = count if (question or replay_speed == 0) else 1
    for i in range(snapshots_to_preload):
        snapshot = replay.next_snapshot()
        if snapshot is None:
            break
        telemetry = snapshot.get("telemetry", {})
        session_info = snapshot.get("session_info", {})
        driver_names = snapshot.get("driver_names", {})
        units = snapshot.get("units", {})
        if driver_names:
            driver_names = {int(k): v for k, v in driver_names.items()}
        state.update(telemetry, session_info, driver_names, units=units)
        snapshots_processed += 1

    # Show final state summary
    player = state.player
    flags = ", ".join(state.flags_list) if state.flags_list else "Green"
    print(
        f"📊 State: Lap {player.lap}, P{player.position}, "
        f"Fuel {player.fuel_pct:.0%}, Flags: {flags}"
    )
    if replay_speed > 0 and not question:
        print(
            f"   Processed {snapshots_processed}/{count} snapshots "
            f"(feeding one every {replay_speed}s)"
        )
    else:
        print(f"   Processed {snapshots_processed} snapshots")
    print()

    # If question provided on command line, ask it and exit
    if question:
        response = ask_engineer(
            state,
            context_builder,
            llm,
            question,
            chat_macros=chat_macros,
            log_label="(replay)",
            iracing_client=None,  # log-only — no sim in replay mode
        )
        if response is None:
            return

        print("=" * 60)
        print("🏁 RACE ENGINEER")
        print("=" * 60)
        print(response)
        print("=" * 60)
        if tts is not None:
            tts.speak_async(response)
        return

    # Interactive mode — type questions, get answers
    print("=" * 60)
    print("🎙️  Interactive Replay Mode")
    print("=" * 60)
    if replay_speed > 0:
        print(f"Timed replay: feeding one snapshot every {replay_speed}s")
        print("Snapshots will advance automatically. Type at any time.")
    print("Type a question and press Enter to query the race engineer.")
    print("Press Enter with no input for a general strategy query.")
    print("Type 'state' to see current race state.")
    print("Type 'depth minimal/medium/full' to change context depth.")
    print("Type 'next' to advance to the next snapshot immediately.")
    print("Type 'voice' to speak your question via microphone.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    # Set up STT for voice input in replay mode
    stt = None
    voice_config = config.get("voice", {})
    if voice_config.get("stt", {}).get("enabled", False):
        try:
            stt = setup_stt(config)
            if stt is not None:
                # Pre-load the STT model — without this, the first
                # 'voice' command triggers a lazy model load that can take
                # 10-30s, exceeding the join timeout below and silently
                # discarding the transcription
                print("  ⏳ Loading STT model for voice input...")
                stt.preload()
        except Exception:
            stt = None

    # For timed replay, feed snapshots in a background thread
    replay_stop = threading.Event()
    # Shared state for dynamic prompt — seed from current state
    prompt_state = {
        "lap": state.player.lap,
        "pos": state.player.position,
        "fuel_pct": state.player.fuel_pct,
        "snapshot_num": snapshots_processed,
    }
    last_progress_idx = [0]  # Track last printed progress milestone

    def _replay_feeder():
        """Background thread that feeds snapshots at the configured interval."""
        while not replay_stop.is_set():
            time.sleep(replay_speed)
            if replay_stop.is_set():
                break
            snapshot = replay.next_snapshot()
            if snapshot is None:
                print("\n  🏁 Replay complete — all snapshots consumed.\n")
                replay_stop.set()
                break
            telemetry = snapshot.get("telemetry", {})
            session_info = snapshot.get("session_info", {})
            driver_names = snapshot.get("driver_names", {})
            units = snapshot.get("units", {})
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            with _state_lock:
                state.update(telemetry, session_info, driver_names, units=units)
                p = state.player
            # Update shared prompt state
            prompt_state["lap"] = p.lap
            prompt_state["pos"] = p.position
            prompt_state["fuel_pct"] = p.fuel_pct
            prompt_state["snapshot_num"] = replay._index
            # Print progress at milestones (every ~20% of total snapshots)
            step = max(1, count // 5)
            if (
                replay._index > 1
                and replay._index % step == 0
                and replay._index != last_progress_idx[0]
            ):
                flags = ", ".join(state.flags_list) if state.flags_list else "Green"
                print(
                    f"  📊 Lap {p.lap}, P{p.position}, Fuel {p.fuel_pct:.0%}, {flags} — {replay._index}/{count}"
                )
                last_progress_idx[0] = replay._index

    if replay_speed > 0:
        feeder_thread = threading.Thread(target=_replay_feeder, daemon=True)
        feeder_thread.start()

    while True:
        try:
            # Build dynamic prompt with current state
            lap = prompt_state["lap"]
            pos = prompt_state["pos"]
            fuel_pct = prompt_state["fuel_pct"]
            snap = prompt_state["snapshot_num"]
            prompt_prefix = f"🏎️ [Lap {lap} P{pos} {fuel_pct:.0%}⛽ {snap}/{count}]"
            user_input = input(f"{prompt_prefix} > ").strip()
            # Strip UTF-8 BOM that PowerShell may prepend when piping input
            if user_input.startswith("﻿"):
                user_input = user_input[1:].strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Replay ended.")
            break

        if not user_input:
            # Empty input = general strategy query (like pressing the query button in live mode)
            user_input = "What's my current strategy?"

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        if user_input.lower() == "voice":
            # Voice input mode — record from microphone, transcribe, query LLM
            if stt is None:
                print(
                    "  ❌ Voice input not available. Install voice deps: uv pip install -e '.[voice]'\n"
                )
                continue
            print("  🎤 Recording... Press Enter to stop.")
            stop_event = threading.Event()
            result = [""]

            def _record():
                try:
                    result[0] = stt.listen_push_to_talk(
                        stop_event,
                        max_duration_s=voice_config.get("trigger", {}).get(
                            "max_record_seconds", 15
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Voice recording/transcription failed (replay mode)"
                    )

            rec_thread = threading.Thread(target=_record)
            rec_thread.start()
            input()  # Wait for Enter to stop
            stop_event.set()
            # Recording stops immediately, but transcription runs inside the
            # same thread and can take a few seconds — join generously so
            # a valid transcription isn't discarded as "no speech"
            rec_thread.join(timeout=120)
            if rec_thread.is_alive():
                print("  ⚠️  Transcription still running — try again shortly.\n")
                continue

            if result[0].strip():
                print(f'  📝 Heard: "{result[0]}"')
                user_input = result[0]
            else:
                print("  ❌ No speech detected.\n")
                continue

        if user_input.lower() == "next":
            # Advance to the next snapshot immediately
            snapshot = replay.next_snapshot()
            if snapshot is None:
                print("  🏁 No more snapshots to replay.\n")
                continue
            telemetry = snapshot.get("telemetry", {})
            session_info = snapshot.get("session_info", {})
            driver_names = snapshot.get("driver_names", {})
            units = snapshot.get("units", {})
            if driver_names:
                driver_names = {int(k): v for k, v in driver_names.items()}
            with _state_lock:
                state.update(telemetry, session_info, driver_names, units=units)
            p = state.player
            print(
                f"  📊 Lap {p.lap}, P{p.position}, "
                f"Fuel {p.fuel_pct:.0%} — "
                f"snapshot {replay._index}/{count}\n"
            )
            continue

        if user_input.lower() == "state":
            # Print current state summary
            p = state.player
            print(
                f"  Track: {state.session.track_name} {state.session.track_config}".strip()
            )
            print(
                f"  Lap: {p.lap}/{p.lap + state.session.laps_remain} | "
                f"Pos: P{p.position} (Class P{p.class_position}) | "
                f"Flags: {flags}"
            )
            print(
                f"  Fuel: {p.fuel_level:.1f}L ({p.fuel_pct:.0%}), "
                f"~{state.fuel_laps_remaining:.1f} laps remaining"
            )
            print(
                f"  Tyres: LF {p.tyres.get('LF', TyreState()).temp_center:.0f}°C / "
                f"RF {p.tyres.get('RF', TyreState()).temp_center:.0f}°C / "
                f"LR {p.tyres.get('LR', TyreState()).temp_center:.0f}°C / "
                f"RR {p.tyres.get('RR', TyreState()).temp_center:.0f}°C"
            )
            if state.nearby_cars:
                print(f"  Nearby: {len(state.nearby_cars)} cars")
                for car in state.nearby_cars[:3]:
                    gap_str = (
                        f"+{car.gap_seconds:.1f}s"
                        if car.gap_seconds >= 0
                        else f"{car.gap_seconds:.1f}s"
                    )
                    print(f"    P{car.position} ({car.driver_name}): {gap_str}")
            print()
            continue

        if user_input.lower().startswith("depth "):
            new_depth = user_input.lower().split()[1]
            if new_depth in ("minimal", "medium", "full"):
                config.setdefault("prompt", {})["context_depth"] = new_depth
                context_builder = ContextBuilder(config)  # Rebuild with new depth
                print(f"  ✅ Context depth set to: {new_depth}\n")
            else:
                print(
                    f"  ❌ Unknown depth: {new_depth}. Use minimal, medium, or full.\n"
                )
            continue

        # It's a question — send to LLM
        print("  ⏳ Asking race engineer...\n")
        response = ask_engineer(
            state,
            context_builder,
            llm,
            user_input,
            chat_macros=chat_macros,
            log_label="(replay interactive)",
            iracing_client=None,  # log-only — no sim in replay mode
        )

        if response:
            print("  " + "=" * 56)
            print("  🏁 RACE ENGINEER")
            print("  " + "=" * 56)
            for line in response.split("\n"):
                print(f"  {line}")
            print("  " + "=" * 56 + "\n")
            if tts is not None:
                tts.speak_async(response)
        else:
            print("  ❌ No response from LLM.\n")

    # Clean up replay feeder thread if running
    if replay_speed > 0:
        replay_stop.set()
        feeder_thread.join(timeout=2)
