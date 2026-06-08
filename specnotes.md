## Architecture: Self-Contained Windows App + LLM API

**Key decision: Everything runs on the Windows iRacing PC.** No Hermes dependency during a race. The spotter is a standalone Python application that reads iRacing shared memory, runs deterministic spotter logic locally, and calls an LLM API only for on-demand strategy questions.

```
iRacing PC (Windows) — ALL-IN-ONE
┌─────────────────────────────────────────────────────┐
│  iracing-spotter.py                                  │
│                                                      │
│  ┌─────────────┐   ┌──────────────┐                 │
│  │ pyirsdk      │──▶│ Spotter logic │──▶ Pre-recorded │
│  │ (shared mem) │   │ (deterministic│    audio out    │
│  └─────────────┘   └──────┬───────┘                 │
│                            │                         │
│  ┌─────────────┐           │ on-demand only           │
│  │ SQLite       │◀─────────┤                         │
│  │ (race state) │           │                         │
│  └─────────────┘           ▼                         │
│                    ┌───────────────┐                  │
│  ┌─────────────┐   │ LLM API call  │  (OpenAI/any)   │
│  │ Mic input    │──▶│ + context from │──────┐         │
│  │ → STT        │   │   SQLite       │      │         │
│  │ (faster-     │   └───────────────┘      ▼         │
│  │  whisper)    │                     ┌──────────┐   │
│  └─────────────┘                     │ TTS call  │   │
│                                      │ (Edge/Piper)│  │
│                                      └─────┬─────┘   │
│                                            │         │
│  ┌─────────────┐                           │         │
│  │ Audio out   │◀──────────────────────────┘         │
│  │ (headphones)│                                     │
│  └─────────────┘                                     │
└─────────────────────────────────────────────────────┘
     │
     │  Separate from:
     ▼
  Discord voice channel (team comms, 7 drivers, unrelated to spotter)
```

### Why self-contained, not Hermes-dependent?
- **No single point of failure.** If Hermes goes down, the spotter still works.
- **Simpler networking.** No cross-machine WebSocket to maintain mid-race.
- **Lower latency.** No LAN transport for audio — everything local.
- **LLM is just an API call.** Send condensed context (~1-2KB), get strategy answer back. No agent needs to be "in the loop" during a race.
- **Offline resilience.** Pre-recorded calls and deterministic logic work without internet. Only LLM strategy queries need connectivity.

### LLM context: what gets sent
NOT raw telemetry. A condensed snapshot only when the driver asks a question:
- Current positions, gaps to nearby cars
- Fuel state, estimated laps remaining
- Tyre status (temps, wear)
- Stint data (lap times trend, degradation)
- Race state (flags, laps remaining, pit window)
- ~1-2KB per call, pennies per race

---

## Language: Python

Phil's more comfortable with Python. Speed of iteration matters more than raw performance for this project — spotter logic is simple conditionals at 30Hz, not compute-bound.

**C# advantages we considered but deprioritised:**
- iRacingSdkWrapper is more mature than pyirsdk
- NAudio is better for Windows audio than Python's options
- Single .exe distribution for teammates
- SimHub plugin compatibility

**Python advantages that won:**
- faster-whisper, edge-tts, SQLite — all pip install away
- Edit-run cycle is instant — critical for tuning cooldowns and thresholds
- ML/TTS ecosystem is native
- Performance isn't the bottleneck — simple comparisons at 30Hz

**Architecture note:** Keep spotter logic (rules, state machine) cleanly separated from the iRacing data layer and audio layer. Spotter rules live in a YAML/JSON config file — proximity thresholds, cooldown timers, call phrases, flag priorities. Makes tuning effortless and would make a theoretical C# port mechanical if ever needed.

---

## Transport Layer Decision: N/A (Self-Contained)

No inter-machine transport needed for the spotter. Everything runs locally on the Windows PC. LLM strategy calls are standard HTTPS to an API provider.

| Option considered | Verdict | Reason |
|-------------------|---------|--------|
| WebSocket to Hermes | Rejected | No need — spotter runs locally, LLM is just an API call |
| MQTT to Hermes | Rejected | Same — no server dependency wanted mid-race |
| UDP forward via SimHub | Rejected | Doesn't work for iRacing |
| **Standalone + LLM API** | **Chosen** | Simpler, more reliable, works offline for core calls |

### SimHub Investigation (rejected path)
SimHub's UDP forwarding does NOT work for iRacing — it only forwards UDP from games that natively output it (F1, pCars, etc.). iRacing uses shared memory, which SimHub reads directly. The third-party SimHubPropertyServer plugin could expose data via TCP, but adds complexity and latency for no benefit over reading shared memory directly. Plus we'd depend on SimHub being installed and running.

**Verdict: Write our own with pyirsdk.** Simpler, direct, full control.

---

## Data Storage: In-Memory + SQLite (Local)

### In-memory (real-time spotter state)
- Latest telemetry frame: positions, speeds, gaps, flags
- Previous frame for delta detection
- Per-driver proximity state, cooldowns
- ~8KB working set. Trivial.

### SQLite (historical / queryable)
Written to on discrete events, not every tick:

| Event | When written | Data |
|-------|-------------|------|
| Lap completion | Each driver crosses line | Lap time, fuel, tyre temps/pressures, position, gaps |
| Flag change | Yellow/green/white/etc. | Flag type, lap, leader |
| Pit stop | Car enters pits | Driver, lap, fuel added, tyres changed |
| Incident | Contact, off-track | Driver, type, location |
| Session end | Race finishes | Final standings, best laps, full results |

~70 bytes per lap per driver. A 60-lap race with 7 drivers = ~30KB total.

SQLite also serves as the LLM context source — when the driver asks "should I pit?", the spotter queries SQLite for stint data, fuel curves, opponent patterns, and includes that in the API call.

**Why not full tick history?** 1,800 frames × 4KB = ~7MB per lap. Unnecessary. Per-lap summaries cover 95% of strategy questions.

---

## Audio Architecture (All Local)

### Pre-recorded calls (safety-critical, ~30-40 fixed phrases)
- Concatenated: `[name].mp3` + `[call].mp3`
- Name files: one per team driver (~7)
- Call files: "car inside", "clear right", "yellow ahead", "green green green", etc.
- Sub-10ms total latency (local playback from RAM)
- Cached at session start

### TTS calls (on-demand strategy questions only)
- Triggered by driver voice input via keyword detection
- Edge TTS (en-GB-ThomasNeural) — requires internet, ~200-500ms generation
- Offline fallback: Piper (CPU, decent quality) or Kokoro (GPU, better quality)
- Acceptable latency for non-urgent info

### Priority
1. **Safety-critical** (flags, proximity) — pre-recorded, immediate
2. **Team-wide** (yellow, green, restart) — pre-recorded, broadcast
3. **Driver-specific** (clearance, pit call) — pre-recorded with name prefix
4. **On-demand** (gaps, fuel, strategy) — LLM + TTS, keyword triggered

---

## STT / Voice Input (Local)

- **faster-whisper** runs locally on the iRacing PC's GPU
- ~100ms transcription time
- Always-listening with keyword detection ("spotter", driver name)
- Or push-to-talk button on wheel (less processing overhead)
- Mic input → VAD (voice activity detection) → keyword check → if match: full STT → intent parsing → action

---

## Latency Budget (estimated)

| Stage | Pre-recorded (local) | TTS (local) | LLM + TTS (API) |
|-------|----------------------|-------------|-----------------|
| Event detection | <1ms | <1ms | <1ms |
| Logic / lookup | 1-2ms | 1-2ms | SQLite query ~5ms |
| Audio generation | 0ms (cached) | 200-500ms | API round-trip ~1-3s |
| Audio playback | 5-10ms | 5-10ms | + TTS 200-500ms |
| **Total** | **~10ms** | **~210-520ms** | **~1.5-4s** |

Pre-recorded path is faster than a human spotter's reaction time. LLM path is acceptable for strategy questions where you're not in traffic.

---

## Spotter Call Categories

### Always call (affects all 7 drivers)
- 🏴 Flags: yellow, green, white, chequered
- 🌧️ Weather: track conditions changing
- 🏁 Start/restart: "Green green green"

### Per-driver only (name-prefixed)
- ⚠️ Proximity: "Phil, car inside" / "Dave, car outside"
- ✅ Clearance: "Phil, clear right" / "Mark, clear left"
- 🔧 Pit: "Dave, box this lap"

### On-demand only (keyword triggered)
- Gaps: "Gap to P3 is X seconds"
- Fuel: "Fuel is X laps remaining"
- Strategy: "Can make it to the end on fuel"

### Suppress (too noisy for 7 drivers)
- General closing rate alerts
- Lapping (beyond leader/blue flag)
- Routine position changes

---

## Config File Structure (YAML/JSON)

Spotter rules, thresholds, and call phrases will live in a config file — no code changes to tune behaviour:

```yaml
# Example spotter config (not final)
proximity:
  inside_threshold_pct: 0.5    # % of lap distance for "car inside"
  closing_threshold_pct: 0.3    # % for "closing fast"
  clearance_threshold_pct: 0.8 # % for "clear"

cooldowns:
  proximity_ms: 3000            # don't repeat same call within 3s
  flag_ms: 5000                 # flag call cooldown
  clearance_ms: 5000

calls:
  car_inside: "car inside"
  car_outside: "car outside"
  clear_right: "clear right"
  clear_left: "clear left"
  yellow: "yellow, yellow, yellow"
  green: "green, green, green"
  three_wide: "three wide, hold your line"

priority:
  - safety_critical    # flags, proximity
  - team_wide          # yellow, green, restart
  - driver_specific    # clearance, pit call
  - on_demand          # gaps, fuel, strategy

voice:
  names:
    phil: "phil.mp3"
    dave: "dave.mp3"
    # ... one per team driver
  prefix_format: "{name}, {call}"  # "Phil, car inside"
```

---

## CrewChief Research Notes

### Jim's voice = human-recorded, NOT TTS
Every default voice line in CrewChief was recorded and edited by Jim Britton (mr_belowski) personally. Thousands of .wav files. This is why "Jim" sounds natural with dry/sarcastic inflection. Alternative voices are either community-recorded or AI-generated via `crew-chief-autovoicepack` (Coqui XTTS).

**Implication:** We can't match Jim's naturalness with TTS. Our advantage is personalisation (driver names, adaptive strategy chat via LLM) and serving all 7 drivers simultaneously.

### crew-chief-autovoicepack
- Uses Coqui XTTS for voice cloning from reference audio
- Reads CrewChief's `phrase_inventory.csv` (thousands of phrases)
- Can clone Jim's voice from his original recordings, or use any voice
- Supports translation via LLM for other languages (German voice pack exists)
- ~2.6GB VRAM for local inference, pack size ~2GB vs Jim's original ~0.5GB
- Also supports ElevenLabs as alternative TTS backend

### CrewChief has NO API / plugin system
- No REST API, no WebSocket, no plugin SDK
- Command macros = static keypress sequences only (voice command → keypresses)
- Can't extend its vocabulary, logic, or audio output
- The only extension point is dropping new .wav files into the sounds folder

### Voice commands CrewChief supports
- Race status: gap ahead/behind, position, fuel level, time remaining
- Car status: damage report, tyre temps, brake temps, fuel usage
- Pit: box this lap, tyres/no tyres, fuel to the end
- Opponent tracking: rival [name], gap to car ahead/behind
- Spotter: automatic (car left/right/clear)
- MFD/pit menu control via keypress macros
- iRacing-specific: black flag status, PSI tyre pressure setup

### Why CrewChief hasn't adopted LLMs
1. **Latency** — spotter calls need <200ms, LLM inference adds 500ms+
2. **Determinism** — must be reliably right every time, no hallucinations
3. **Cost** — every telemetry tick hitting an LLM API is expensive
4. **State-machine approach works** — telemetry event → exact voice line is instant and reliable

**Our approach:** pre-recorded for safety-critical, LLM for on-demand strategy only. The LLM never touches real-time spotter calls.

---

## TTS Voice Landscape

### Edge TTS vs Azure Speech Services
Same engine, different front door. Edge TTS is the reverse-engineered free path. Azure is the paid official API.

| | Edge TTS | Azure Speech Services |
|---|---------|----------------------|
| Cost | Free | $16/M chars after 5M free |
| API key | None needed | Azure subscription required |
| SLA | None | Yes |
| SSML support | Limited | Full |
| Offline | No | No |
| Risk | MS could block endpoint | Stable, supported |

### Local TTS Options (for offline resilience)

| Engine | Truly Local? | Quality | Notes |
|--------|-------------|---------|-------|
| Edge TTS | ❌ Cloud | Good | Free, same engine as Azure, needs internet |
| Piper | ✅ CPU | Decent | ONNX models, fast, good offline option |
| Kokoro | ✅ CPU/GPU | Good | Newer OSS, very good quality for size |
| NeuTTS | ✅ CPU | OK | GGUF models |
| Windows SAPI | ✅ Local | Poor | Built into Windows, robotic |

**Recommendation:** Edge TTS as primary (quality), Piper as offline fallback (reliability). Pre-recorded calls work regardless of internet.

---

## Data Size Reference

| What | Size | Notes |
|------|------|-------|
| One telemetry frame (all 30 cars) | ~4KB | Position, speed, fuel, tyres per car |
| One telemetry frame (7 team cars) | ~1KB | Filtered to relevant cars only |
| Working memory (current + prev frame) | ~8KB | Bugger all |
| Full race raw data (60 laps, 7 cars) | ~3-7MB | If keeping per-frame (unlikely needed) |
| SQLite per-lap summaries (60 laps, 7 cars) | ~30KB | Sufficient for all strategy queries |
| Pre-recorded call MP3s (~40 phrases) | ~2-5MB | Cached at session start |
| LLM context snapshot | ~1-2KB | Per API call, pennies per race |

---

## Implementation Steps (when ready)
1. Write iracing-spotter.py — reads shared memory via pyirsdk, deterministic spotter logic, pre-recorded audio playback
2. Define YAML/JSON config — proximity thresholds, cooldown timers, call phrases, flag priorities
3. Add SQLite persistence for race state (lap times, fuel, flags, pit stops)
4. Add mic input + faster-whisper STT for keyword detection and driver questions
5. Add LLM API call for on-demand strategy questions (context from SQLite)
6. Add Edge TTS (primary) + Piper (offline fallback) for variable content
7. Pre-generate name + call MP3 files
8. Test with iRacing practice session
9. Tune call frequency, cooldowns, priority rules

---

## Open Questions
- Mic input — push-to-talk (wheel button) or always-listen with keyword detection?
- How to handle multiple team drivers hearing different calls? (Separate audio streams, or shared?)
- Audio routing — should the spotter go through the same headset as Discord, or a separate output?
- Pre-recorded phrases — record ourselves, generate via TTS, or borrow from CrewChief's GPL audio?
- Name prefix format: "Phil, car inside" vs "Car inside, Phil" — which is clearer in a loud race?
- Which LLM provider for strategy calls? OpenAI cheapest/most capable, but requires internet. Local LLM on GPU would add latency.
