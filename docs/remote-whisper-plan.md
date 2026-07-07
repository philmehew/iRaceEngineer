# Plan: Move Whisper to Remote Server

## Context

The iRaceEngineer client currently runs `faster-whisper` locally for speech-to-text. During races, iRacing and Whisper compete for GPU resources, causing CUDA timeouts (up to 49s observed) and forcing CPU fallback. Moving Whisper to a dedicated remote server eliminates this contention entirely, giving faster and more reliable transcription.

## Key Decisions

- **Server location**: Same LAN (no auth/TLS needed — simple HTTP)
- **Server project**: Separate repo (`whisper-server/`), not in this repo
- **This repo**: Only the client-side changes — remote transcription mode in `stt_client.py`

## Architecture

```
Racing PC (this repo)                     LAN Server (separate repo)
┌──────────────────┐                        ┌──────────────────┐
│ sounddevice      │                        │ faster-whisper   │
│   ↓ record      │   HTTP POST /transcribe │   ↓ transcribe   │
│ WAV encode       │ ──────────────────────→ │ model (loaded)   │
│   ↓ send         │ ←────────────────────── │   ↓              │
│                  │   JSON {"text": "..."}   │ result           │
└──────────────────┘                        └──────────────────┘
```

**Local mode** (unchanged): mic → sounddevice → numpy → faster-whisper → text
**Remote mode**: mic → sounddevice → numpy → WAV bytes → HTTP POST → server → text
**Remote with fallback**: tries remote first, falls back to local on any failure

## Files to Create/Modify

### 1. NEW: Separate `whisper-server` repo (NOT in this repo)

A separate Python project with its own `pyproject.toml`. This plan defines the **API contract** the client expects so both sides stay in sync.

**API contract (both repos must implement):**

- `POST /transcribe` — accepts WAV audio multipart upload (`audio` file field) + query params (`language`, `vad_filter`), returns JSON `{"text": "...", "language": "...", "duration": 3.14}`
- `GET /health` — returns JSON `{"status": "ok", "model": "...", "device": "..."}`

**Server implementation (in the separate repo):**
- FastAPI + uvicorn, `faster-whisper` for transcription
- Config via env vars: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_PORT` (default 8399), `WHISPER_HOST` (default 0.0.0.0)
- No auth (LAN-only) — simplified for trusted network use
- Model loaded at startup via FastAPI lifespan, kept in memory
- WAV decoding: accepts int16 WAV, converts to float32 numpy for faster-whisper
- Run: `python -m whisper_server` or `uvicorn whisper_server:app --host 0.0.0.0 --port 8399`

### 2. MODIFY: `stt_client.py`

Refactor to support three modes: `local`, `remote`, `remote_with_fallback`.

**New config keys** (under `voice.stt`):
- `mode` — `"local"` (default), `"remote"`, `"remote_with_fallback"`
- `remote_url` — e.g. `"http://192.168.0.200:8399"` (LAN, no auth needed)
- `remote_timeout` — HTTP timeout in seconds (default 5.0)

**Key changes:**
- Extract current `transcribe()` body → `_transcribe_local()` (no logic changes)
- Add `_transcribe_remote()` — encodes audio as WAV, POSTs to server, returns text
- Add `audio_to_wav_bytes()` helper — converts float32 numpy to int16 WAV bytes
- Add `RemoteSTTError` exception class
- Refactor `transcribe()` to dispatch: remote → `_transcribe_remote()`, local → `_transcribe_local()`, fallback → try remote then local
- Update `is_available` — in remote mode, always returns True (no faster-whisper needed locally)
- Update `preload()` — in remote mode, calls `_check_remote_health()` instead of loading local model
- Add `_check_remote_health()` — probes `GET /health` on startup, logs result

### 3. MODIFY: `config.yaml`

Add new keys under `voice.stt`:

```yaml
voice:
  stt:
    enabled: true
    mode: "remote"                           # "local", "remote", "remote_with_fallback"
    # Local settings (used in local mode or as fallback)
    model: "small"
    device: "cuda"
    compute_type: "float16"
    # Remote settings
    remote_url: "http://192.168.0.200:8399"  # Whisper server URL (LAN, no auth)
    remote_timeout: 5.0                       # Seconds
    # Shared settings
    input_device: 1
    input_gain: 1.0
    vad_filter: true
    language: "en"
```

### 4. MODIFY: `main.py`

Minimal changes — the `STTClient` interface stays the same:
- Update startup log to show mode ("remote", "remote with local fallback", "local")
- Gate CUDA DLL check on local/fallback mode only (skip for pure remote)

### 5. MODIFY: `test_stt.py`

Add `--mode` and `--remote-url` CLI args to test remote mode:
```bash
python test_stt.py --mode remote --remote-url http://192.168.0.200:8399 --push-to-talk
```

### 6. MODIFY: `pyproject.toml`

- Add explicit `httpx>=0.27.0` to main dependencies (already transitive via `openai`, but now used directly)
- The `whisper-server` project is a **separate repo** with its own `pyproject.toml` (FastAPI, uvicorn, python-multipart, faster-whisper)

## Fallback Behavior

| Scenario | `remote` | `remote_with_fallback` | `local` |
|---|---|---|---|
| Server unreachable | Error, return "" | Warning, fall back to local | N/A |
| Server timeout | Error, return "" | Warning, fall back to local | N/A |
| Server returns error | Error, return "" | Warning, fall back to local | N/A |
| Local CUDA fails | N/A | CPU fallback | CPU fallback |
| Both fail | Return "" | Return "" | Return "" |

## Latency Estimate (over LAN)

- 5s audio → 240KB int16 WAV → uploads in ~2ms on Gigabit LAN
- Whisper "small" on dedicated GPU: ~200-500ms
- Total round-trip: ~250-600ms vs 500-2000ms+ local (with GPU contention from iRacing)

## Verification

**Client (this repo):**
1. Test remote mode: `python test_stt.py --mode remote --remote-url http://<server>:8399 --push-to-talk`
2. Test fallback: kill the server, verify `remote_with_fallback` falls back to local
3. Test live mode: `python main.py --voice` with `mode: remote` in config
4. Run existing tests: `pytest tests/`

**Server (separate repo):**
1. Start the server: `uvicorn whisper_server:app --host 0.0.0.0 --port 8399`
2. Test health: `curl http://localhost:8399/health`
3. Test transcription: `curl -X POST http://localhost:8399/transcribe -F "audio=@test.wav"`
4. Test with client: configure `remote_url` and run from racing PC
