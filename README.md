# iRaceEngineer

Real-time iRacing data collection with on-demand LLM race engineering. Collects telemetry from iRacing via shared memory, maintains an in-memory model of the current race state, and sends a condensed snapshot to an OpenAI-compatible LLM when you press F9 — or speak your question via push-to-talk (F10) and hear the response spoken aloud.

## What It Does

- **Collects** real-time telemetry from iRacing at ~30Hz via pyirsdk (323+ telemetry variables + YAML session info)
- **Maintains** an in-memory race state model — positions, gaps, fuel, tyres, lap trends, weather, engine health, damage, push-to-pass
- **Condenses** that state into a ~1-2KB context prompt, filtered by configurable depth (minimal / medium / full)
- **Sends** the prompt to any OpenAI-compatible LLM endpoint when you press F9 (text query) or hold F10 (voice query)
- **Parses** optional `[ACTION]` directives in the LLM response (pit this lap, add fuel, change tyres) — logged in dry-run mode, executable when enabled
- **Speaks** LLM responses aloud via Piper TTS, and **listens** for voice questions via Whisper STT (optional, fully local)
- **Spotter** — real-time audio calls for car proximity using pre-recorded WAV files (car left, car right, three wide, clear). Deterministic local logic, no LLM involved
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
        ├──────────────────────────┐
        │                          │
        ▼ (on button press)       ▼ (every tick)
 context_builder.py   ← condenses     spotter.py ← deterministic audio calls
        │              state →         │            (car left/right, three wide, clear)
        ▼              configurable    ▼
 llm_client.py        depth prompt   Pre-recorded WAV files (audio/)
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
# Normal mode — connect to iRacing, F9 for text query, F10 for voice query
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
| `medium` | + gaps, tyre temps, flags, engine warnings, track conditions | ~250 bytes |
| `full` | + engine health, tyre life & odometers, damage, proximity, session config, lap trends, nearby cars, pit status, weather, shift lights | ~1-2 KB |

### Example: Full Context

```
Driver: Mayhem. Race: Silverstone Circuit Arena Grand Prix, Lap 4/32771, P16 (Class P16)
Flags: Green
Track: 5.8km | 18 turns | pit limit 60.0kph
Session: FIXED SETUP | incidents: unlimited | fast repairs: unlimited | tank: 22L
Weather: Track 40.6°C | Air 18.3°C | track Damp | wind 0.89m/s
Engine: Oil 69.6°C | OilP 3.40bar | Water 70.5°C | 13.80V | Manifold 0.99bar

Your car:
  Position: P16 (Class P16)
  Fuel: 11.66L (53%), ~22.0 laps remaining
  Tank capacity: 22.0L (53% full)
  Fuel burn rate: 34.38 L/hr
  Tyres:
    LF: 79.9°C | 124.11PSI | life 98.4% | 18800km
    RF: 73.8°C | 124.11PSI | life 98.5% | 18700km
    LR: 81.7°C | 124.11PSI | life 98.5% | 18800km
    RR: 76.6°C | 124.11PSI | life 98.4% | 18800km
  Brake bias: 56.00%
  Damage: incidents 16x (team 16x)
  Shift: shift at 6000rpm | 40% throttle
  Proximity: car LEFT | +15.57s ahead | -7.86s behind

Pit: Pits open | Fast repair available | Tire sets available: 1

Nearby:
  P15 (Car #5): -6.223s, last lap 2:28.872
  P17 (Car #3): -0.001s, last lap 2:28.872
  P14 (Car #12): +0.003s, last lap 2:28.339
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

The client captures 323+ telemetry variables from iRacing shared memory plus session info (YAML). Key categories sent to the LLM:

| Category | Example Variables | Use for LLM |
|----------|-------------------|-------------|
| Car state | Speed, Gear, RPM, Throttle, Brake, IsOnTrack | Driving style, car status |
| Engine health | OilTemp, OilPress, WaterTemp, EngineWarnings, Voltage, ManifoldPress | Engine health monitoring |
| Fuel | FuelLevel, FuelLevelPct, FuelUsePerHour, tank capacity | Pit strategy |
| Tyres | Per-corner temps (3 zones), pressures, life %, odometers | Degradation model |
| Brakes | Brake line pressure per corner, brake bias, ABS | Brake management |
| Damage | Incident counts, weight penalty, fast repairs, repair time needed | Impact assessment |
| Position & proximity | CarIdxPosition, CarDistAhead/Behind, CarLeftRight, tow time | Gap analysis |
| Laps | Lap times, deltas, best laps, trend | Pace analysis |
| Weather | TrackTemp, TrackWetness (Dry/Damp/Wet/VeryWet), AirTemp, WindVel | Weather strategy |
| Session config | Fixed setup, incident limit, tank capacity, shift RPMs, pit speed limit | Rules awareness |
| Pit | PitsOpen, PitstopActive, FastRepairAvailable, tyre sets | Pit window |
| Push-to-pass | P2P_Status, P2P_Count | Overtake opportunities |
| Shift | ShiftIndicatorPct, PlayerCarSLShiftRPM | Shift lights |
| G-forces | LatAccel, LongAccel, VertAccel | Driving analysis |

## Voice Input/Output

iRaceEngineer supports push-to-talk voice input (Whisper) and spoken responses (Piper TTS). Both are optional — install the voice extras to enable them.

### Install Voice Dependencies

```bash
uv sync --extra voice
```

This installs `faster-whisper`, `piper-tts`, and `sounddevice`. For GPU acceleration (optional):

```bash
# Whisper GPU (CUDA 12 + cuDNN 9)
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

# Piper GPU (onnxruntime with CUDA)
uv pip install onnxruntime-gpu
```

### Download a Voice Model

```bash
# Download the default British male voice (race engineer vibe)
python -m piper.download_voices en_GB-alan-medium --data-dir voices

# Or download via the test script
python test_tts.py --download en_GB-alan-medium
```

Voice models are stored in `voices/` (gitignored). Available voices: [piper-samples](https://rhasspy.github.io/piper-samples/)

### Test TTS Standalone

```bash
# Speak text through your default audio device
python test_tts.py "Box this lap, tyres are gone. Add sixty litres."

# List available audio output devices
python test_tts.py --list-devices

# List available voice aliases and downloaded models
python test_tts.py --list-voices

# Play through a specific device (use index from --list-devices)
python test_tts.py --device 5 "Hello"

# Use a voice alias (short name)
python test_tts.py --voice alan "Box this lap"
python test_tts.py --voice cori "Box this lap"
python test_tts.py --voice cori-high "Box this lap"
python test_tts.py --voice northern "Box this lap"

# Use a full model name
python test_tts.py --model en_US-lessac-medium "Hello"

# Write to WAV file instead of playing (useful over SSH)
python test_tts.py --wav out.wav "Box this lap, tyres are gone"
python test_tts.py --voice northern --wav out.wav "Box this lap"

# Read text from a file
python test_tts.py --file response.txt

# Adjust volume
python test_tts.py --volume 0.5 "Quiet please"
```

### Test STT Standalone

```bash
# Record for 5 seconds, transcribe, print
python test_stt.py

# List available audio input devices
python test_stt.py --list-devices

# Use a specific microphone (use index from --list-devices)
python test_stt.py --device 1

# Push-to-talk: press Enter to start, Enter again to stop
python test_stt.py --push-to-talk

# Change Whisper model size
python test_stt.py --model tiny     # fastest, least accurate
python test_stt.py --model medium  # better accuracy, slower

# Boost mic input (useful for quiet microphones)
python test_stt.py --gain 2.0 --push-to-talk

# Record for longer
python test_stt.py --duration 10
```

### Test Wheel Button Discovery

Use `test_wheel.py` to identify your steering wheel's device and button indices for push-to-talk:

```bash
# List connected controllers
python test_wheel.py --list

# Watch for button presses (press buttons on your wheel to identify them)
python test_wheel.py

# Monitor only a specific device
python test_wheel.py --device 0

# Also show analog axis movements
python test_wheel.py --watch-axes
```

When you press a button, the script prints the device index, button number, and a suggested `config.yaml` snippet.

### Voice in Live Mode

In live mode, two triggers are available:

- **F9** — Text trigger (no question, general strategy query) — always keyboard
- **F10 or wheel button** — Push-to-talk voice input (hold to speak, release to transcribe)

By default, voice push-to-talk uses the F10 key. You can also use a button on your steering wheel:

1. Install pygame: `uv sync --extra wheel`
2. Discover your wheel's device and button indices:
   ```bash
   python test_wheel.py --list      # List connected controllers
   python test_wheel.py              # Press buttons to see their indices
   ```
3. Configure in `config.yaml`:
   ```yaml
   voice:
     trigger:
       method: "wheel_button"
       device_index: 0    # From test_wheel.py --list
       button_index: 4    # From pressing buttons in test_wheel.py
       push_to_talk: true
       max_record_seconds: 15
   ```

Hold the wheel button to speak, release to transcribe. The response is both printed and spoken aloud through Piper TTS.

### Voice in Replay Mode

In interactive replay mode:

- Type **`voice`** to enter voice input mode — records from your microphone, transcribes, and sends to LLM
- LLM responses are spoken aloud via TTS (when enabled)

### CLI Flags

```bash
# Force-enable voice for this run (overrides config)
python main.py --voice

# Force-disable voice for this run
python main.py --no-voice
```

### Audio Device Selection

To route TTS output to a specific device (e.g. a headset or virtual audio cable), set the device index or name in `config.yaml`:

```yaml
voice:
  tts:
    output_device: 5               # Numeric index or device name string
  stt:
    input_device: 1                # Numeric index or device name string
```

Find device indices with:
```bash
python test_tts.py --list-devices   # Output devices (for TTS + spotter)
python test_stt.py --list-devices   # Input devices (for STT mic)
```

Set to `null` (or omit) to use the system default.

### Volume & Mic Gain

All audio levels are configurable via multipliers in `config.yaml`:

```yaml
voice:
  stt:
    input_gain: 1.0    # Mic input gain (1.0 = no boost, try 1.5–2.0 for quiet mics)
  tts:
    volume: 1.0        # TTS output volume (1.0 = normal, 0.5 = half, 2.0 = double)
spotter:
  volume: 1.0          # Spotter call volume (1.0 = normal, independent of TTS)
```

- **`input_gain`** — multiplies mic audio before sending to Whisper. Useful if your mic is quiet and transcription is poor. Try `1.5` or `2.0`.
- **`volume`** (TTS) — multiplies LLM spoken responses. Set to `0.5` to make them quieter, `2.0` for louder.
- **`volume`** (Spotter) — same multiplier for car proximity calls. Independent of TTS volume.

### Preventing Discord Pick-up (Virtual Audio Cable)

If you use Discord voice chat while racing, teammates will hear the race engineer's audio through your mic. Turning the volume down helps but doesn't fully fix it — Discord's noise suppression is designed for steady background noise, not intermittent speech.

The solution is to route race engineer audio through a **virtual audio cable** so you hear it but Discord doesn't pick it up:

1. **Install VB-Audio Virtual Cable** (free): [vb-audio.com/Cable](https://vb-audio.com/Cable/)

2. **Configure iRaceEngineer** to send TTS and spotter audio to the virtual cable:
   ```yaml
   voice:
     tts:
       output_device: "CABLE Input"     # Virtual cable output
   spotter:
     output_device: "CABLE Input"       # Same device for spotter calls
   ```
   Use the exact device name string, or the numeric index from `python test_tts.py --list-devices`.

3. **Route the virtual cable to your headset** so you can still hear it:
   - Windows Sound settings → Recording tab → right-click "CABLE Output" → Properties
   - Listen tab → check "Listen to this device" → select your real headset/headphones
   - This plays the virtual cable's audio through your headset — you hear it, Discord's mic doesn't

4. **Keep Discord's input device** as your real microphone (not the virtual cable)

Result: you hear the race engineer and spotter through your headset, but Discord only transmits your voice — not the race engineer.

## Spotter (Car Proximity Audio)

iRaceEngineer includes a **local, deterministic spotter** that plays pre-recorded audio calls when cars appear alongside you. No LLM is involved — this is pure edge-detection logic running at the telemetry poll rate (~30Hz).

### How It Works

The spotter reads iRacing's `CarLeftRight` telemetry value every tick:
- **0** = no car alongside
- **1** = car on your left
- **2** = car on your right
- **3** = car on both sides (three wide)

It detects **transitions** (not steady state), so you only hear a call when a car appears or clears — not every tick. Cooldown timers prevent repeated calls.

| Transition | Call | Audio File |
|------------|------|------------|
| Car appears on left (0→1, 2→3) | `car_left` | `audio/carleft.wav` |
| Car appears on right (0→2, 1→3) | `car_right` | `audio/carright.wav` |
| Both sides appear from none (0→3) | `three_wide` | `audio/carthreewide.wav` |
| Any side clears (1→0, 2→0, 3→0, 3→1, 3→2) | `clear` | `audio/carclear.wav` |

### Audio Files

Place WAV files (mono, any sample rate) in the `audio/` directory and configure paths in `config.yaml`. Audio is loaded into memory at startup for low-latency playback. Missing files are logged as warnings but don't crash the app.

### Cooldowns

- **Proximity calls** (car left, car right, three wide): 3 second cooldown between repeat calls
- **Clearance calls** (clear): 5 second cooldown between repeat calls

### Disabling the Spotter

Set `spotter.enabled: false` in `config.yaml`, or the spotter won't activate.

## Configuration

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
  method: "keyboard"    # "keyboard" or "wheel_button"
  key: "f9"

# Voice input/output (optional: uv sync --extra voice)
voice:
  stt:
    enabled: true
    model: "small"                    # Whisper model size
    device: "cuda"                     # cuda or cpu
    compute_type: "float16"
    input_device: null                # null = system default
    input_gain: 1.0                   # Mic input gain multiplier (1.0 = no boost, 2.0 = 2x louder)
    vad_filter: true
    language: "en"
  tts:
    enabled: true
    model: "en_GB-alan-medium"         # Piper voice model
    voice_dir: "voices"
    use_cuda: true
    output_device: null               # null = system default
    volume: 1.0
    sentence_silence: 0.2
  trigger:
    method: "keyboard"           # "keyboard" or "wheel_button"
    voice_key: "f10"             # Keyboard key (when method: keyboard)
    push_to_talk: true
    max_record_seconds: 15
    device_index: null           # Joystick device index (when method: wheel_button)
    button_index: null           # Button index (when method: wheel_button)

# Spotter — real-time audio calls for car proximity (local, no LLM)
spotter:
  enabled: true
  audio_paths:
    car_left: "audio/carleft.wav"           # Car appears on left
    car_right: "audio/carright.wav"          # Car appears on right
    three_wide: "audio/carthreewide.wav"     # Cars on both sides simultaneously
    clear: "audio/carclear.wav"              # Car clears from alongside
  cooldowns:
    proximity_ms: 3000                        # Don't repeat car left/right within 3s
    clearance_ms: 5000                        # Don't repeat clear calls within 5s
  output_device: null                         # null = system default
  volume: 1.0                                 # Volume multiplier (independent of TTS)

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

# Timed replay — feed one snapshot every N seconds (state evolves like a real race)
# Use floats for sub-second intervals:
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41 --replay-speed 0.1   # ~3 min for 1920 snapshots
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41 --replay-speed 1     # ~32 min (roughly real-time at 1Hz)
python main.py --replay tests/sample_data/session_2026-06-08_13-38-41 --replay-speed 10    # slow: one snapshot every 10s
```

Enters interactive mode where you can type questions and press Enter. The prompt shows live race state:

```
🏎️ [Lap 4 P16 53%⛽ 500/1920] > Should I pit?
```

Interactive commands:
- Type a question and press Enter to query the race engineer
- Press Enter with no input for a general strategy query
- `voice` — speak your question via microphone (requires voice extras)
- `state` — show current race state summary
- `depth minimal/medium/full` — change context depth
- `next` — advance to the next snapshot immediately
- `quit` or Ctrl+C — exit

With `--replay-speed > 0`, snapshots feed automatically in the background and the prompt updates with live Lap, Position, Fuel, and snapshot progress. Milestone progress is printed every ~20% of total snapshots. Without it (or `--replay-speed 0`), all snapshots are loaded instantly and you interact with the final state.

Works over SSH — no GUI or physical keyboard needed.

### LLM Response Evaluation

Batch-test the race engineer by replaying a session and asking 50 questions at even intervals:

```bash
# Run 50 questions against the Silverstone race data
python tests/eval_llm_responses.py --count 50 --output logs/llm_eval.md

# Fewer questions, different context depth
python tests/eval_llm_responses.py --count 10 --depth medium --output logs/eval_medium.md
```

This picks evenly-spaced snapshots from the race, asks a rotating set of race engineering questions (fuel, tyres, strategy, engine health, etc.), and logs the full prompt + response to a markdown file. Each entry has an expandable `<details>` section showing exactly what context was sent to the LLM.

Options:
- `--count N` — number of questions to ask (default 50)
- `--depth minimal/medium/full` — context depth (default: full)
- `--output PATH` — output markdown file (default: `logs/llm_eval.md`)
- `--session PATH` — session data directory (default: the Silverstone session)

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
| `main.py` | Entry point — CLI args, poll loop, keyboard trigger, interactive replay, voice |
| `iracing_client.py` | pyirsdk wrapper — telemetry, session info, pit commands |
| `race_state.py` | In-memory model — positions, gaps, fuel, tyres, engine health, lap trends |
| `context_builder.py` | Condenses state → LLM prompt (3 depth levels, format helpers) |
| `llm_client.py` | OpenAI-compatible API caller (works with any `/v1` endpoint) |
| `action_executor.py` | Parses `[ACTION]` directives from LLM responses |
| `spotter.py` | Deterministic car proximity calls — edge-detect CarLeftRight transitions, play WAV files |
| `stt_client.py` | Speech-to-text — Whisper (faster-whisper) + mic capture (sounddevice) |
| `tts_client.py` | Text-to-speech — Piper TTS + audio playback (sounddevice) |
| `capture.py` | Record/replay telemetry JSON for testing |
| `test_stt.py` | Standalone STT test — record + transcribe without iRacing |
| `test_tts.py` | Standalone TTS test — speak text without iRacing |
| `test_wheel.py` | Standalone wheel button discovery — find device/button indices for push-to-talk |
| `tests/test_modules.py` | Unit tests — RaceState, ContextBuilder, ActionExecutor, TelemetryReplay |
| `tests/test_spotter.py` | Unit tests — ProximityDetector state machine, SpotterAudioPlayer, Spotter coordinator |
| `tests/eval_llm_responses.py` | Batch LLM evaluation — asks 50 questions against replay data, logs results |

## Roadmap

This is the first working slice of the iRaceEngineer spotter system. Planned next steps from the [spec notes](specnotes.md):

- [ ] **SQLite persistence** — store per-lap data for historical queries
- [ ] **Pre-recorded audio** — spotter calls for flags, more clearance phrases
- [x] **TTS output** — Piper TTS for spoken LLM responses (local, offline)
- [x] **Mic input / STT** — faster-whisper for push-to-talk voice questions
- [ ] **Multi-driver audio** — name-prefixed calls for 7 drivers
- [x] **Spotter proximity logic** — deterministic 30Hz calls for car proximity (car left/right, three wide, clear)
- [ ] **Action execution** — enable real `[ACTION]` commands to iRacing
- [ ] **Team-aware prompts** — label teammates explicitly in nearby cars section, show per-teammate data, separate team section in full depth

## License

Private project — not yet licensed for distribution.
