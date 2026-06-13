# iRaceEngineer

Real-time iRacing data collection with on-demand LLM race engineering. Reads telemetry from iRacing shared memory via pyirsdk, maintains race state, and sends condensed context to an OpenAI-compatible LLM when triggered.

## Architecture

```
iRacing (shared memory) → iracing_client.py → race_state.py → context_builder.py → llm_client.py → response
                                                        ↓                      ↓
                                                  spotter.py             action_executor.py (dry_run by default)
                                                 (local audio)                    ↓
                                                                        tts_client.py (Piper TTS → speakers)

F10 key or wheel button (hold) → stt_client.py (mic → Whisper) → transcribed text → context_builder → LLM
```

- **iracing_client.py** — pyirsdk wrapper, reads 327 telemetry variables + session info, exposes pit commands
- **race_state.py** — DriverState (universal per-car class), SessionState, RaceState with per-lap history
- **context_builder.py** — condenses state into ~0.2-2KB prompt at 3 depth levels (minimal/medium/full)
- **llm_client.py** — OpenAI-compatible API caller, works with Ollama Cloud, OpenAI, LM Studio, etc.
- **action_executor.py** — parses [ACTION] directives from LLM responses, dry_run by default (v1)
- **spotter.py** — deterministic real-time audio calls for car proximity (car left/right, three wide, clear) and car-behind-closing alerts using pre-recorded WAV files; no LLM involved
- **stt_client.py** — speech-to-text via faster-whisper + sounddevice mic capture, push-to-talk
- **tts_client.py** — text-to-speech via Piper TTS + sounddevice playback, configurable output device
- **capture.py** — record/replay telemetry JSON for testing without iRacing
- **main.py** — entry point with CLI: live, --capture, --replay, --generate-samples, --voice; includes WheelButtonListener for steering wheel PTT

## Key Design Decisions

- **DriverState is universal** — used for player, teammates, and nearby cars. Player-only fields (fuel, tyre temps/wear, brakes) are left at defaults for other cars
- **Team detection** — auto-detect by iRacing TeamName, with config fallback for explicit username/car number lists
- **Config-driven context depth** — minimal (~200B), medium (~500B), full (~1-2KB), controlled by config.yaml
- **Action directives** — LLM can include `[ACTION] pit_this_lap`, `[ACTION] add_fuel: 60` etc. v1 logs but doesn't execute
- **Testable without iRacing** — `--generate-samples` creates fake data, `--replay <dir>` feeds it through the pipeline
- **Voice is optional** — voice deps in `[voice]` extra (`uv sync --extra voice`), graceful degradation if not installed
- **Wheel button is optional** — pygame in `[wheel]` extra (`uv sync --extra wheel`), for push-to-talk via steering wheel button instead of keyboard F10
- **Voice tested independently** — `test_stt.py` and `test_tts.py` are standalone scripts for testing each component
- **Spotter is local and deterministic** — reads CarLeftRight telemetry at 30Hz, plays pre-recorded WAV files on transitions (car appears/clears alongside). Also detects car-behind-closing using lap-time delta comparison (not noisy CarDistBehind derivative). No LLM involved. Edge-detection with cooldown timers prevents repeated calls. Uses sounddevice.OutputStream (not sd.play) to avoid conflicting with TTS.
- **Still-there reminder** — when a car has been alongside continuously for more than `still_there_delay_ms` (default 5000ms), plays `carstillthere.wav` as a reminder. Repeats every `still_there_cooldown_ms` (default 10000ms) while the car remains alongside. Resets when the car clears.

## Tech Stack

- Python 3.13, managed by uv
- pyirsdk (1.3.5) — iRacing shared memory
- openai (2.x) — LLM API client
- pyyaml — config
- keyboard — F9/F10 trigger (needs admin on Windows)
- pygame (2.6+) — steering wheel button trigger for push-to-talk (optional `[wheel]` extra)
- faster-whisper (1.2+) — local speech-to-text via Whisper (optional `[voice]` extra)
- piper-tts (1.4+) — local text-to-speech (optional `[voice]` extra)
- sounddevice — mic capture + audio playback (optional `[voice]` extra)
- Config in config.yaml, LLM endpoint configurable for Ollama Cloud / OpenAI / local

## Running

```bash
python main.py                          # Live mode — connect to iRacing, F9 to query LLM, F10 for voice
python main.py --capture                # Record live telemetry to JSON
python main.py --replay <dir>           # Replay captured data
python main.py --generate-samples       # Generate fake Spa 24h stint data
python main.py --depth minimal          # Override context depth
python main.py --question "Should I pit?"  # Ask a specific question
python main.py --voice                  # Force-enable voice input/output
python main.py --no-voice               # Force-disable voice input/output
python test_tts.py "Box this lap"       # Test TTS standalone
python test_stt.py --push-to-talk       # Test STT standalone
python test_wheel.py                    # Discover wheel button device/index for push-to-talk
python test_wheel.py --list             # List connected controllers
```

## Current LLM Config

- Endpoint: Ollama Cloud (https://ollama.com/v1)
- Model: ministral-3:14b-cloud
- API key stored directly in config.yaml (api_key_env field — client auto-detects direct keys vs env var names)

## Team Config

- 7 drivers: Patrik Farsang, Wayne Smith8, Andre Groove, Dan Golden, Dave Cartnerdobbs, Liam Biggs, David Barlow
- Auto-detect by iRacing TeamName with config fallback
- All 7 use the same DriverState class, player gets full fuel/tyre data

## Sample Data

Located in `tests/sample_data/` — 10 snapshots simulating a Spa 24h race stint (laps 1-10), with progressive tyre deg, fuel burn, and realistic opponent lap times.

## File Locations

- Config: `config.yaml`
- Wheel button test: `test_wheel.py`
- Spotter audio: `audio/` (carleft.wav, carright.wav, carthreewide.wav, carclear.wav, carstillthere.wav, carbehindclosing.wav, plus flag/penalty/fuel/pit WAVs)
- Sample data: `tests/sample_data/session_*`
- Tests: `tests/test_modules.py`, `tests/test_spotter.py` (ProximityDetector, CarBehindTracker, Spotter coordinator, SpotterAudioPlayer)
- Plan file: `C:\Users\phil\.claude\plans\can-you-review-these-pure-crayon.md`
- Spec notes: `specnotes.md`
