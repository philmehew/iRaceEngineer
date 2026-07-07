# Plan: Remote Piper TTS Server

Move Piper TTS inference to a remote GPU server using **wyoming-piper** (the official Piper server) with its built-in HTTP bridge. The client gets a new `RemoteTTSClient` that calls the HTTP API and plays audio locally. Config toggles between local/remote mode.

---

**Steps**

### Phase 1: Remote Server Setup

1. **Create `docker-compose.tts.yml`** — Docker Compose file for the remote GPU server with two services:
   - `wyoming-piper`: runs `rhasspy/wyoming-piper` with `--voice`, `--use-cuda`, `--data-dir`, GPU device passthrough, exposes port 10200 (Wyoming TCP protocol)
   - `wyoming-http`: runs the wyoming HTTP bridge (`wyoming.http.tts_server`) connecting to the piper service, exposes port 5000 (HTTP REST API)
   - Volume mount for voice model data persistence

2. **Create `docs/remote-tts.md`** — Setup guide documenting:
   - Docker commands, GPU setup (`--gpus all` or specific device)
   - API: `POST /api/tts?text=...&voice=...` → `audio/wav` response
   - Network/firewall requirements
   - How to download voice models

### Phase 2: Client-Side Changes

3. **Update `config.yaml`** — Add under `voice.tts`:
   ```yaml
   tts:
     mode: "local"              # "local" or "remote"
     # ... existing local config ...
     remote:
       base_url: "http://192.168.0.200:5000"   # wyoming HTTP bridge URL
       timeout: 10.0                            # seconds
       voice: null                              # override voice name sent to server (null = use server default)
   ```

4. **Add `RemoteTTSClient` class to `tts_client.py`** — *parallel with step 5*
   - `__init__`: reads `voice.tts.remote` config (base_url, timeout, voice)
   - `speak(text)`: preprocess → HTTP POST `{base_url}/api/tts?text=...` → receive WAV → decode with `wave` module → play via `sd.play()` + `sd.wait()`
   - `speak_async(text)`: wraps `speak()` in daemon thread (same pattern as local)
   - `synthesize(text)`: HTTP POST → decode WAV → return `(audio_float32, sample_rate)` (same interface as local)
   - `is_available`: lightweight health check (GET to base_url)
   - `preload()`: no-op (remote server handles model loading)
   - Error handling: timeout, connection errors → log warning, return gracefully
   - Reuses `preprocess_for_tts()` for text normalization

5. **Add `create_tts_client(config)` factory to `tts_client.py`** — *parallel with step 4*
   - Reads `voice.tts.mode` from config
   - Returns `RemoteTTSClient` or `TTSClient` accordingly
   - Graceful fallback: if mode is "remote" but server unreachable → log warning, return None

6. **Update `main.py`** — *depends on steps 4-5*
   - Replace `TTSClient(config)` with `create_tts_client(config)` in both `run_live_mode()` and `run_replay_mode()`
   - No other changes — both classes expose same interface (`speak()`, `speak_async()`, `is_available`, `preload()`)

7. **Update `pyproject.toml`** — *parallel with steps 4-6*
   - Move `piper-tts` from main `dependencies` to `[voice]` optional dependencies (remote mode doesn't need it locally)
   - Add `requests` explicitly to main deps (currently only transitive via openai)

8. **Update `test_tts.py`** — *depends on step 4*
   - Add `--remote` flag to test against remote TTS endpoint
   - Add `--base-url` flag to specify server URL
   - Test both local and remote modes

### Phase 3: Verification

9. **Test local mode** — `voice.tts.mode: local` should behave identically to current
10. **Test remote mode** — Start Docker stack, run iRaceEngineer with `mode: remote`, verify audio plays
11. **Test fallback** — `mode: remote` with unreachable server → graceful TTS disable with warning log

---

**Relevant files**
- `c:\iRaceEngineer\tts_client.py` — Add `RemoteTTSClient` class, `create_tts_client()` factory; reuse `preprocess_for_tts()`, `_resolve_output_device()` pattern
- `c:\iRaceEngineer\config.yaml` — Add `mode` and `remote` section under `voice.tts`
- `c:\iRaceEngineer\main.py` — Replace `TTSClient(config)` with `create_tts_client(config)` in `run_live_mode()` and `run_replay_mode()`
- `c:\iRaceEngineer\pyproject.toml` — Move `piper-tts` to optional deps, add `requests`
- `c:\iRaceEngineer\test_tts.py` — Add remote test flags
- `c:\iRaceEngineer\docker-compose.tts.yml` — NEW: Docker Compose for remote Piper server
- `c:\iRaceEngineer\docs\remote-tts.md` — NEW: Remote server setup docs

---

**Verification**
1. Run `python main.py --generate-samples` with `mode: local` — TTS should work as before
2. Start Docker stack (`docker compose -f docker-compose.tts.yml up`), run with `mode: remote` — audio should play from remote server
3. Set `mode: remote` with unreachable `base_url` — should log warning and disable TTS gracefully (no crash)
4. Run `python test_tts.py --remote --base-url http://remote:5000 "Box this lap"` — should play audio from remote server
5. Verify `piper-tts` is no longer required when `mode: remote` (uninstall it, confirm no import errors)

---

**Decisions**
- **Server**: wyoming-piper with HTTP bridge — official, maintained, GPU support, Docker-ready
- **Protocol**: HTTP REST (`POST /api/tts`) not Wyoming TCP — simpler client, no async dependency
- **Audio format**: WAV (what the HTTP bridge returns) — decoded with stdlib `wave`, no new deps
- **Fallback**: Remote failure → disable TTS entirely (don't silently fall back to local; user chose remote for a reason)
- **Config**: `voice.tts.mode` toggle; remote config nested under `voice.tts.remote`
- **No streaming**: HTTP bridge returns complete WAV. Fine for short race engineer responses (~20 words)

---

**Further Considerations**
1. **Latency**: LAN round-trip adds ~1-5ms vs local Piper synthesis ~300-1000ms. Negligible. WAN would need testing.
2. **Auth**: wyoming HTTP bridge has no auth. For LAN this is fine. For WAN, add a reverse proxy (nginx) with basic auth and an `api_key` field in remote config.
3. **Multiple voices**: The `voice` query param allows per-request voice selection. Could extend config to support voice switching without server restart.
