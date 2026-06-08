# iRaceEngineer

Real-time iRacing data collection with on-demand LLM race engineering. Reads telemetry from iRacing shared memory via pyirsdk, maintains race state, and sends condensed context to an OpenAI-compatible LLM when triggered.

## Architecture

```
iRacing (shared memory) → iracing_client.py → race_state.py → context_builder.py → llm_client.py → response
                                                                                ↓
                                                                         action_executor.py (dry_run by default)
```

- **iracing_client.py** — pyirsdk wrapper, reads 327 telemetry variables + session info, exposes pit commands
- **race_state.py** — DriverState (universal per-car class), SessionState, RaceState with per-lap history
- **context_builder.py** — condenses state into ~0.2-2KB prompt at 3 depth levels (minimal/medium/full)
- **llm_client.py** — OpenAI-compatible API caller, works with Ollama Cloud, OpenAI, LM Studio, etc.
- **action_executor.py** — parses [ACTION] directives from LLM responses, dry_run by default (v1)
- **capture.py** — record/replay telemetry JSON for testing without iRacing
- **main.py** — entry point with CLI: live, --capture, --replay, --generate-samples

## Key Design Decisions

- **DriverState is universal** — used for player, teammates, and nearby cars. Player-only fields (fuel, tyre temps/wear, brakes) are left at defaults for other cars
- **Team detection** — auto-detect by iRacing TeamName, with config fallback for explicit username/car number lists
- **Config-driven context depth** — minimal (~200B), medium (~500B), full (~1-2KB), controlled by config.yaml
- **Action directives** — LLM can include `[ACTION] pit_this_lap`, `[ACTION] add_fuel: 60` etc. v1 logs but doesn't execute
- **Testable without iRacing** — `--generate-samples` creates fake data, `--replay <dir>` feeds it through the pipeline

## Tech Stack

- Python 3.13, managed by uv
- pyirsdk (1.3.5) — iRacing shared memory
- openai (2.x) — LLM API client
- pyyaml — config
- keyboard — F9 trigger (needs admin on Windows)
- Config in config.yaml, LLM endpoint configurable for Ollama Cloud / OpenAI / local

## Running

```bash
python main.py                          # Live mode — connect to iRacing, F9 to query LLM
python main.py --capture                # Record live telemetry to JSON
python main.py --replay <dir>           # Replay captured data
python main.py --generate-samples       # Generate fake Spa 24h stint data
python main.py --depth minimal          # Override context depth
python main.py --question "Should I pit?"  # Ask a specific question
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
- Sample data: `tests/sample_data/session_*`
- Plan file: `C:\Users\phil\.claude\plans\can-you-review-these-pure-crayon.md`
- Spec notes: `specnotes.md`
