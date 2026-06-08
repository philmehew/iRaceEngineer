# iRaceEngineer

Real-time iRacing data collection with on-demand LLM race engineering. Collects telemetry from iRacing via shared memory, maintains an in-memory model of the current race state, and sends a condensed snapshot to an OpenAI-compatible LLM when you press a button — getting back concise, actionable strategy advice.

## What It Does

- **Collects** real-time telemetry from iRacing at ~30Hz via pyirsdk (327 telemetry variables + YAML session info)
- **Maintains** an in-memory race state model — positions, gaps, fuel, tyres, lap trends, weather, damage, push-to-pass
- **Condenses** that state into a ~1-2KB context prompt, filtered by configurable depth (minimal / medium / full)
- **Sends** the prompt to any OpenAI-compatible LLM endpoint when you press a button
- **Parses** optional `[ACTION]` directives in the LLM response (pit this lap, add fuel, change tyres) — logged in dry-run mode, executable when enabled
- **Records** and **replays** telemetry snapshots for testing without iRacing running

## Architecture

```
iRacing (shared memory)
        │
        ▼
 iracing_client.py   ← continuous 30Hz poll via pyirsdk
        │
        ▼
 race_state.py        ← in-memory model: positions, fuel, tyres, laps, trends
        │
        ▼ (on button press)
 context_builder.py   ← condenses state → configurable depth prompt
        │
        ▼
 llm_client.py        ← OpenAI-compatible API call (Ollama Cloud, OpenAI, LM Studio, etc.)
        │
        ▼
 action_executor.py   ← parses [ACTION] directives (dry_run by default)
```

## Quick Start

### Prerequisites

- Python 3.13+
- iRacing installed and running (for live mode)
- An OpenAI-compatible LLM API key (Ollama Cloud, OpenAI, etc.)

### Install

```bash
cd iRaceEngineer
uv sync
```

### Configure

Edit `config.yaml` to set your LLM endpoint:

```yaml
llm:
  base_url: "https://api.ollama.com/v1"   # Change for OpenAI, LM Studio, local Ollama
  api_key_env: "OLLAMA_API_KEY"            # Environment variable holding your API key
  model: "ministral-3:14b-cloud"            # Or gpt-4o-mini, etc.
```

Set your API key as an environment variable:

```bash
# PowerShell
$env:OLLAMA_API_KEY = "your-api-key-here"

# Or CMD
set OLLAMA_API_KEY=your-api-key-here
```

### Run

```bash
# Normal mode — connect to iRacing, press F9 to query LLM
python main.py

# Replay mode — test with captured data, no iRacing needed
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41

# Capture mode — record live telemetry to JSON files
python main.py --capture

# Generate sample data for testing
python main.py --generate-samples

# Save real iRacing data to the sample data folder
python main.py --save-samples

# Override context depth
python main.py --depth minimal
```

## Context Depths

The amount of race data sent to the LLM is controlled by `context_depth` in `config.yaml`:

| Depth | Includes | Approx. Size |
|-------|----------|-------------|
| `minimal` | Position, fuel, laps remaining | ~100 bytes |
| `medium` | + gaps, tyre temps, flags | ~200 bytes |
| `full` | + lap trends, nearby cars, pit status, weather, damage, push-to-pass | ~800 bytes |

### Example: Full Context

```
Race: Circuit de Spa-Franchamps Grand Prix, Lap 10/60, P4 (Class P2)
Flags: Green
Track: 27°C, Air: 22°C

Your car:
  Position: P4 (Class P2)
  Fuel: 72.0L (65%), ~18.9 laps remaining
  Fuel burn rate: 28.5 L/hr
  LF: 97°C | 26.5 PSI | wear 10.0%
  RF: 99°C | 26.3 PSI | wear 9.0%
  LR: 105°C | 24.8 PSI | wear 14.0%
  RR: 107°C | 24.5 PSI | wear 15.0%
  Incidents: 0 (team: 2)
  Push-to-pass: 3 remaining
  Last 5 laps: 1:33.400, 1:33.550, 1:33.700, 1:33.850, 1:34.000 (fuel: 3.8L each)
  Trend: slowing 0.15s/lap

Pit: Pits open | Fast repair available | Tire sets available: 3

Nearby:
  P3 (Smith): -0.020s, last lap 1:32.400, P2P available
  P2 (Dave): 0.000s, last lap 1:32.100
```

## Team Detection

iRaceEngineer identifies your teammates using two strategies that work **additively** — both are checked, and results are unioned:

### Auto-detect (default)

iRacing assigns each driver a `TeamName` in session info. If you and your teammates all set the same TeamName in the iRacing UI, they're found automatically — no config needed. Enabled by default (`team.auto_detect: true`).

### Config fallback

If TeamName isn't set or you want to be explicit, list usernames or car numbers in `config.yaml`. These are matched **in addition to** auto-detect, not as a replacement:

```yaml
team:
  auto_detect: true
  teammates:
    - "Patrik Farsang"
    - "Wayne Smith8"
    - "Andre Groove"
  car_numbers: []        # Alternative: identify by car number
```

### Team data in the LLM prompt

- **Team incidents** — the prompt includes `team_incidents` (an iRacing-provided aggregate of your whole team's incident count), shown as `Incidents: 0 (team: 2)` in full depth mode
- **Nearby cars** — teammates appear in the nearby cars section based on position proximity, but are not explicitly labelled as teammates in the prompt (the system prompt tells the LLM to "use driver names when referring to teammates")
- **DriverState is universal** — the same `DriverState` class is used for the player, teammates, and opponents. Player-only fields (fuel, tyre temps/wear, brakes) default to zero for non-player cars

> **Note:** Team car indices are detected and stored internally, but are not yet used to give teammates special treatment in the prompt (e.g., richer data, explicit team labels, separate team section). This is planned for a future update.

## Action Directives

The LLM can include optional `[ACTION]` directives in its response:

```
Box this lap — tyres are past their window and you have 8 laps of fuel.
[ACTION] pit_this_lap
[ACTION] add_fuel: 60
[ACTION] change_tyres
```

In v1, actions are **logged but not executed** (`actions.enabled: false` in config). This ensures the architecture is in place without risking unintended pit commands. To enable real execution:

```yaml
actions:
  enabled: true
  allowed_actions:
    - pit_this_lap
    - add_fuel
    - change_tyres
    - clear_penalty
```

Available actions (mapped to pyirsdk `PitCommandMode`):

| Action | iRacing Command | Description |
|--------|----------------|-------------|
| `pit_this_lap` | Request pit stop | Pit on the current lap |
| `add_fuel: N` | Set fuel amount | Add N litres at next pit stop |
| `change_tyres` | Request tyre change | Change tyres at next pit stop |
| `clear_penalty` | Clear penalty | Clear a penalty |

## iRacing SDK Data

The script captures all 327 telemetry variables from iRacing shared memory. Key categories:

| Category | Example Variables | Use for LLM |
|----------|-------------------|-------------|
| Car state | Speed, Gear, RPM, Throttle, Brake | Driving style analysis |
| Fuel | FuelLevel, FuelLevelPct, FuelUsePerHour | Pit strategy |
| Tyres | LF/RF/LR/RR temps, pressures, wear (per zone) | Degradation model |
| Brakes | Brake line pressure per corner, ABS | Brake management |
| Position | CarIdxPosition, CarDistAhead/Behind | Gap analysis |
| Laps | Lap times, deltas, best laps | Trend analysis |
| Weather | TrackTemp, TrackWetness, AirTemp, Precipitation | Weather strategy |
| Pit | PitsOpen, PitstopActive, FastRepairAvailable | Pit window |
| Push-to-pass | P2P_Status, P2P_Count | Overtake opportunities |
| Damage | Incident counts, weight penalty | Impact assessment |
| Session | Flags, laps remaining, time remaining | Race state |

## Configuration

Full `config.yaml` reference:

```yaml
# iRacing connection
iracing:
  tick_rate_hz: 30

# LLM API (any OpenAI-compatible endpoint)
llm:
  base_url: "https://api.ollama.com/v1"
  api_key_env: "OLLAMA_API_KEY"
  model: "ministral-3:14b-cloud"
  max_tokens: 300
  temperature: 0.3

# Action execution (dry_run by default)
actions:
  enabled: false
  allowed_actions:
    - pit_this_lap
    - add_fuel
    - change_tyres
    - clear_penalty

# Button trigger
trigger:
  method: "keyboard"    # "keyboard" or "wheel_button" (future)
  key: "f9"

# Prompt configuration
prompt:
  system: |
    You are a race engineer for a sim racing team...
  context_depth: "full"  # "minimal", "medium", or "full"
  include_lap_history: 5
  include_nearby_cars: 3

# State tracking
state:
  track_lap_times: true
  track_fuel: true
  track_tyre_temps: true

# Capture/replay test mode
capture:
  interval_ms: 1000
  output_dir: "./tests/sample_data"

# LLM query logging
logging:
  llm_query_log_dir: "logs"              # Directory for log files
  llm_query_log_file: "llm_queries.log" # Log filename
  llm_query_max_bytes: 1000000          # Rotate at 1 MB
  llm_query_backup_count: 5             # Keep 5 rotated files
```

## LLM Query Logging

Every time the LLM is queried (via F9 keypress, `--question`, or interactive replay), the full prompt and response are written to a local log file. This is useful for debugging context quality, reviewing what the LLM received, and auditing responses.

Log files are written to `logs/llm_queries.log` by default, with automatic rotation (1 MB per file, 5 backups). The `logs/` directory is gitignored.

### Example log entry

```
2026-06-08 14:30:05
--- LLM QUERY ---
Depth: full
Question: Should I pit now?

=== PROMPT SENT ===
[
  {
    "role": "system",
    "content": "You are a race engineer for a sim racing team..."
  },
  {
    "role": "user",
    "content": "Race: Circuit de Spa-Francorchamps Grand Prix, Lap 10/60, P4 (Class P2)\nFlags: Green\n..."
  }
]

=== RESPONSE ===
Based on current tyre wear and fuel levels, I recommend pitting this lap...
--- END ---
```

Log entries are tagged by source:
- `--- LLM QUERY ---` — live F9 keypress
- `--- LLM QUERY (replay) ---` — one-shot `--question` with `--replay`
- `--- LLM QUERY (replay interactive) ---` — typed question in interactive replay mode

## Testing Without iRacing

### Generate Sample Data

```bash
python main.py --generate-samples
```

Creates simulated telemetry (10 laps of a Spa 24h race stint) in `tests/sample_data/`.

### Replay Captured Data

```bash
# Replay with looping (default) — loads all snapshots instantly
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41

# Replay once and stop
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41 --no-loop

# Timed replay — feed one snapshot every 10 seconds (state evolves like a real race)
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41 --replay-speed 10
```

Enters interactive mode where you can type questions and press Enter. Press Enter with no input for a general strategy query (same as pressing F9 in live mode). Type `state` to see current race state, `next` to advance to the next snapshot immediately, `quit` to exit.

With `--replay-speed N`, snapshots are fed automatically every N seconds in the background. The race state evolves over time so you can ask the LLM questions at different points in the stint. Without it (or `--replay-speed 0`), all snapshots are loaded instantly and you interact with the final state.

Works over SSH — no GUI or physical keyboard needed.

### Capture Live Data

```bash
# Capture to a custom directory
python main.py --capture --capture-interval 1000
```

Records telemetry snapshots every 1 second while iRacing is running. Useful for building a library of race scenarios for testing.

### Save Real Data as Samples

```bash
# Save real iRacing data directly to tests/sample_data/
python main.py --save-samples
```

Same as `--capture` but always saves to the `tests/sample_data/` folder (from config), making it easy to capture real race data for replay and testing without specifying a directory.

## Module Reference

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point — CLI args, poll loop, keyboard trigger |
| `iracing_client.py` | pyirsdk wrapper — telemetry, session info, pit commands |
| `race_state.py` | In-memory model — positions, gaps, fuel, tyres, lap trends |
| `context_builder.py` | Condenses state → LLM prompt (3 depth levels) |
| `llm_client.py` | OpenAI-compatible API caller (works with any `/v1` endpoint) |
| `action_executor.py` | Parses `[ACTION]` directives from LLM responses |
| `capture.py` | Record/replay telemetry JSON for testing |

## Roadmap

This is the first working slice of the iRaceEngineer spotter system. Planned next steps from the [spec notes](specnotes.md):

- [ ] **SQLite persistence** — store per-lap data for historical queries
- [ ] **Pre-recorded audio** — spotter calls for flags, proximity, clearance
- [ ] **TTS output** — Edge TTS (primary) + Piper (offline fallback) for variable content
- [ ] **Mic input / STT** — faster-whisper for keyword detection and driver questions
- [ ] **Multi-driver audio** — name-prefixed calls for 7 drivers
- [ ] **Spotter proximity logic** — deterministic 30Hz calls for safety-critical alerts
- [ ] **Action execution** — enable real `[ACTION]` commands to iRacing
- [ ] **Team-aware prompts** — label teammates explicitly in nearby cars section, show per-teammate data, separate team section in full depth

## License

Private project — not yet licensed for distribution.
