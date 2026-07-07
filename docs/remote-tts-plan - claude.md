Ready for review
Select text to add comments on the plan
Plan: Remote Piper TTS Support
Context
Piper TTS currently runs locally on the sim rig, loading the .onnx model into GPU memory and synthesising audio in-process. The goal is to move Piper inference to a dedicated remote GPU server so the sim rig only handles audio playback. This frees local GPU resources for iRacing and Whisper (STT), and enables using more powerful Piper voice models that wouldn't fit in local VRAM.

The user doesn't yet have the remote server set up, so the plan covers both the server implementation and the client changes.

Design Decisions
Decision	Choice	Rationale
HTTP client library	httpx (not requests)	Already installed as transitive dep of openai. No new dependency.
Server framework	Custom FastAPI (not wyoming-piper)	Simpler, self-contained, per-request sentence_silence and volume. No Docker bridge component needed.
Client architecture	Modify TTSClient in place (not separate class)	Single class with mode attribute. Callers unchanged. Factory not needed — __init__ reads mode from config.
Audio format	WAV response	Self-describing (sample rate in header). Client parses with stdlib wave module — same pattern as spotter._load_wav().
Volume	Applied server-side	Sent as parameter in request. Server uses SynthesisConfig.volume. Client receives ready-to-play audio.
Auth	Optional X-API-Key header	Off by default (LAN/SSH tunnel). --api-key CLI flag on server, remote_api_key in client config.
Files to Change
1. NEW: tts_server.py — Remote Piper TTS server
Standalone FastAPI + uvicorn server that runs on the GPU machine. Not imported by the main app.

Startup: Load PiperVoice once via lifespan handler (not deprecated @app.on_event). Resolve model path from --model + --voice-dir (same search logic as client's _find_model_path).
POST /tts: Accepts JSON {text, sentence_silence, volume, speaker_id}. Synthesizes via PiperVoice.synthesize() with SynthesisConfig. Inserts silence bytes between chunks (matching Piper's own http_server.py pattern). Returns audio/wav with X-Sample-Rate header.
GET /health: Returns {status, model, sample_rate, cuda}.
Auth: If --api-key is set, validate X-API-Key header on all requests via FastAPI dependency. 401 on mismatch.
CLI args: --host, --port, --model, --voice-dir, --no-cuda, --sentence-silence, --volume, --api-key, --log-level
2. NEW: Dockerfile and docker-compose.yml
Server deployment files at project root.

Dockerfile: Python 3.13-slim, install espeak-ng (Piper dependency), install piper-tts + fastapi + uvicorn + numpy, copy tts_server.py and voices/, expose port 5000.
docker-compose.yml: Single tts-server service with GPU device reservation, volume mount for voices/, configurable model/port/api-key via environment variables.
3. MODIFY: tts_client.py — Add remote mode
Add mode attribute ("local" or "remote", default "local"). In remote mode:

__init__: Read remote_url, remote_timeout, remote_api_key from config. Set _voice = None, _voice_loaded = True, _load_lock = None. Lazy-import httpx and raise ImportError if missing.
_synthesize_remote(): POST to {remote_url}/tts with JSON body {text, sentence_silence, volume}. Parse WAV response via wave.open(io.BytesIO(...)). Convert to float32 numpy (same pattern as spotter._load_wav()). Return (audio, sample_rate). Handle ConnectError, TimeoutException, HTTPStatusError with logged warnings and empty audio return.
speak(): Dispatch to _synthesize_remote() or _synthesize_local() based on mode. Playback always local via sd.play(). Volume NOT re-applied for remote mode (server already applied it).
synthesize(): Same dispatch.
_synthesize_local(): Extract current local synthesis logic from speak() into this method.
_load_voice(): Early return for remote mode.
is_available: For remote mode, GET /health with 3-second timeout. For local, existing import check.
preload(): For remote mode, GET /health with 5-second timeout, log model info. For local, existing model loading.
speak_async(): No changes needed — delegates to speak() which handles both modes.
4. MODIFY: config.yaml — Add remote TTS settings
Add under voice.tts:

voice:
  tts:
    enabled: true
    mode: "local"                       # "local" (default) or "remote"
    # Local mode settings (used when mode=local):
    model: "en_GB-northern_english_male-medium"
    voice_dir: "voices"
    use_cuda: true
    # Remote mode settings (used when mode=remote):
    remote_url: "http://192.168.0.115:5000"  # GPU server URL
    remote_timeout: 10.0                      # Seconds to wait for synthesis
    remote_api_key: null                      # Optional API key (null = no auth)
    # Common settings (both modes):
    output_device: null
    volume: 1.0
    sentence_silence: 0.2
5. MODIFY: main.py — Mode-aware startup
Minimal changes. TTSClient(config) already reads mode from config. Two awareness notes:

_ensure_cuda_dlls() is still needed (STT uses CUDA locally). No change to this call.
tts.preload() works for both modes. No change needed.
The is_available check at construction time now makes a network call for remote mode (3s timeout). If the server is unreachable, TTS is gracefully disabled.
6. MODIFY: test_tts.py — Remote testing flags
Add CLI arguments:

--remote URL — Set mode to remote with given URL
--remote-timeout SECONDS — Remote request timeout (default 10.0)
--remote-api-key KEY — API key for remote server
--health URL — Check health endpoint and exit
7. MODIFY: pyproject.toml — Add tts-server dependency group
[project.optional-dependencies]
tts-server = [
    "piper-tts>=1.4.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "numpy>=2.4.6",
]
Keep piper-tts and sounddevice in main dependencies (local mode still works). Do NOT add httpx — it's a transitive dep of openai.

Security
Default deployment: LAN-only, no auth. Bind to 0.0.0.0:5000, firewall to sim rig IP.
Recommended: SSH tunnel (ssh -L 5000:localhost:5000 gpu-server). Set remote_url: "http://localhost:5000". All traffic encrypted, no extra auth needed.
API key: Optional --api-key on server, remote_api_key in client config. Simple X-API-Key header check.
No TLS in server: Terminate TLS at reverse proxy or use SSH tunnel.
Verification
Local mode regression: mode: local → identical behavior to current code
Remote mode: Start python tts_server.py, set mode: remote, run iRaceEngineer → audio plays
Server unreachable: Set mode: remote with bad URL → graceful disable, no crash
Health check: curl http://gpu-server:5000/health → JSON with model, sample_rate, cuda
API key: Start server with --api-key secret, verify 401 without key, 200 with X-API-Key: secret
test_tts.py: python test_tts.py --remote http://gpu-server:5000 "Box this lap" → audio plays
SSH tunnel: ssh -L 5000:localhost:5000 gpu-server, set remote_url: "http://localhost:5000" → works
