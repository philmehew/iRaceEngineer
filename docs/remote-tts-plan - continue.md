# Plan: Remote Piper TTS Server

## Current State

Piper TTS currently runs **locally** in `tts_client.py`:
- `PiperVoice.load()` loads the `.onnx` model into memory (optionally with CUDA)
- `synthesize()` / `speak()` run inference locally and play via `sounddevice`
- The model files live in `voices/` directory
- `test_tts.py` provides standalone testing with `--wav` output option

The project also has a precedent for remote API calls — `llm_client.py` uses the OpenAI-compatible API pattern (base URL + API key + model name).

## Recommended Remote Architecture: HTTP REST API

**Best practice** for serving Piper on a remote GPU server is a lightweight HTTP API server using **FastAPI** + **uvicorn**. This is the standard Python approach for ML model serving:

- **FastAPI** — async, typed, auto-docs, production-grade
- **uvicorn** — ASGI server, handles concurrent requests efficiently
- **Piper** — loaded once at startup, kept in GPU memory for fast inference
- **WAV/PCM response** — raw audio bytes returned directly (no file I/O overhead)

This mirrors how the LLM client already works (HTTP POST → response), keeping the architecture consistent.

## Changes Required

### 1. New File: `tts_server.py` (Remote Server)

A standalone FastAPI server that:
- Loads the Piper model at startup (with CUDA if available)
- Exposes `POST /tts` endpoint accepting JSON `{"text": "...", "speaker_id": 0, "sentence_silence": 0.2}`
- Returns audio as WAV with content type headers
- Exposes `GET /health` for monitoring
- Supports configurable host/port/model/volume via CLI args and env vars
- Includes graceful shutdown

```python
"""
Piper TTS Remote Server — serves text-to-speech over HTTP.

Runs on a GPU server, loads the Piper model once at startup,
and synthesizes text to audio on demand.

Usage:
    python tts_server.py                                    # Default: localhost:5000, en_GB-alan-medium
    python tts_server.py --host 0.0.0.0 --port 8080        # Listen on all interfaces
    python tts_server.py --model en_GB-cori-high            # Use a different voice
    python tts_server.py --no-cuda                           # Force CPU inference

Endpoints:
    POST /tts   — Synthesize text to audio
    GET  /health — Health check
"""

import argparse
import io
import logging
import time
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger("tts_server")

app = FastAPI(title="Piper TTS Server", version="1.0.0")

# Global voice instance — loaded once at startup
_voice = None
_voice_config = {"model_name": "", "sample_rate": 22050}


class TTSRequest(BaseModel):
    text: str
    speaker_id: int | None = 0
    sentence_silence: float | None = 0.2
    volume: float | None = 1.0


@app.on_event("startup")
async def load_model():
    """Load Piper model at startup."""
    global _voice
    # ... load PiperVoice.load(model_path, use_cuda=use_cuda) ...
    # Store sample_rate in _voice_config


@app.post("/tts")
async def synthesize(request: TTSRequest):
    """Synthesize text to audio and return as WAV."""
    global _voice
    if _voice is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # ... synthesize with _voice.synthesize(text) ...
    # ... collect chunks, concatenate, apply volume ...
    # ... write to WAV in memory, return bytes ...

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sample_rate)},
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if _voice is not None else "loading",
        "model": _voice_config["model_name"],
        "sample_rate": _voice_config["sample_rate"],
    }


if __name__ == "__main__":
    # ... argparse for --host, --port, --model, --voice-dir, --no-cuda ...
    uvicorn.run(app, host=args.host, port=args.port)
```

### 2. Modified File: `tts_client.py` (Client — Dual Mode)

The client needs to support both **local** and **remote** modes, selected via config:

```yaml
voice:
  tts:
    enabled: true
    mode: "remote"              # "local" (default) or "remote"
    # --- Local mode settings (unchanged) ---
    model: "en_GB-alan-medium"
    voice_dir: "voices"
    use_cuda: true
    # --- Remote mode settings (new) ---
    remote_url: "http://gpu-server:5000"
    remote_timeout: 10.0        # seconds
    # --- Common settings ---
    output_device: null
    volume: 1.0
    sentence_silence: 0.2
```

Key changes to `TTSClient`:

```python
class TTSClient:
    def __init__(self, config: dict):
        tts_config = config.get("voice", {}).get("tts", {})
        self.mode = tts_config.get("mode", "local")  # "local" or "remote"

        if self.mode == "remote":
            # Remote mode — no local model needed
            self.remote_url = tts_config.get("remote_url", "http://localhost:5000")
            self.remote_timeout = tts_config.get("remote_timeout", 10.0)
            self._voice = None  # Not used in remote mode
            self._voice_loaded = True  # Skip local loading
        else:
            # Local mode — existing behavior
            self.model_name = tts_config.get("model", "en_GB-alan-medium")
            self.voice_dir = tts_config.get("voice_dir", "voices")
            self.use_cuda = tts_config.get("use_cuda", True)
            self._voice = None
            self._voice_loaded = False
            self._load_lock = threading.Lock()

        # Common settings (both modes)
        self.volume = tts_config.get("volume", 1.0)
        self.sentence_silence = tts_config.get("sentence_silence", 0.2)
        self._output_device = tts_config.get("output_device", None)

    def speak(self, text: str):
        """Synthesize text and play through audio device."""
        if not text.strip():
            return
        text = preprocess_for_tts(text)

        if self.mode == "remote":
            audio, sample_rate = self._synthesize_remote(text)
        else:
            audio, sample_rate = self._synthesize_local(text)

        if audio.size == 0:
            return

        # ... existing playback logic (sounddevice) ...
        # This stays the same — we just get audio bytes differently

    def _synthesize_remote(self, text: str) -> tuple[np.ndarray, int]:
        """Call remote TTS server and return audio array + sample rate."""
        import requests

        try:
            response = requests.post(
                f"{self.remote_url}/tts",
                json={
                    "text": text,
                    "sentence_silence": self.sentence_silence,
                    "volume": self.volume,
                },
                timeout=self.remote_timeout,
            )
            response.raise_for_status()

            sample_rate = int(response.headers.get("X-Sample-Rate", 22050))

            # Parse WAV bytes back to numpy array
            audio = self._parse_wav_response(response.content)
            return audio, sample_rate

        except requests.exceptions.ConnectionError:
            logger.error(f"TTS server unreachable at {self.remote_url}")
            return np.array([], dtype=np.float32), 22050
        except requests.exceptions.Timeout:
            logger.error(f"TTS server timeout ({self.remote_timeout}s)")
            return np.array([], dtype=np.float32), 22050

    def _parse_wav_response(self, wav_bytes: bytes) -> np.ndarray:
        """Parse WAV bytes from server response into float32 numpy array."""
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # Convert to float32 (same logic as SpotterAudioPlayer._load_wav)
        if sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            logger.error(f"Unsupported sample width: {sample_width}")
            return np.array([], dtype=np.float32)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to numpy array without playing."""
        text = preprocess_for_tts(text)

        if self.mode == "remote":
            return self._synthesize_remote(text)
        else:
            return self._synthesize_local(text)

    def _synthesize_local(self, text: str) -> tuple[np.ndarray, int]:
        """Local Piper synthesis (existing logic, extracted from speak/synthesize)."""
        self._load_voice()
        if self._voice is None:
            return np.array([], dtype=np.float32), 22050
        # ... existing synthesize logic ...

    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        if self.mode == "remote":
            # Check if remote server is reachable
            try:
                import requests
                resp = requests.get(f"{self.remote_url}/health", timeout=3)
                return resp.status_code == 200
            except Exception:
                return False
        else:
            try:
                from piper import PiperVoice  # noqa: F401
                return True
            except ImportError:
                return False

    def preload(self):
        """Pre-load model or check remote server connectivity."""
        if self.mode == "remote":
            # Check remote server is reachable
            try:
                import requests
                resp = requests.get(f"{self.remote_url}/health", timeout=5)
                if resp.status_code == 200:
                    info = resp.json()
                    logger.info(
                        f"TTS remote server connected: model={info.get('model', 'unknown')}"
                    )
                else:
                    logger.warning(f"TTS remote server returned status {resp.status_code}")
            except Exception as e:
                logger.warning(f"TTS remote server unreachable: {e}")
        else:
            # Existing local preload logic
            logger.info("Pre-loading Piper voice model...")
            # ...
```

### 3. Modified File: `pyproject.toml` (Dependencies)

Add server dependencies as a new optional group:

```toml
[project.optional-dependencies]
voice = [
    "faster-whisper>=1.2.0",
    "piper-tts>=1.4.0",
    "sounddevice>=0.5.0",
]
tts-server = [
    "piper-tts>=1.4.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "numpy>=2.4.6",
]
# ... existing groups ...
```

The `requests` library is already a transitive dependency of `openai`, so no need to add it explicitly.

### 4. Modified File: `test_tts.py` (Remote Testing)

Add `--remote` flag to test against a remote server:

```python
# Add to argparse:
parser.add_argument(
    "--remote",
    metavar="URL",
    help="Test remote TTS server (e.g. http://gpu-server:5000)",
)
parser.add_argument(
    "--remote-timeout",
    type=float,
    default=10.0,
    help="Timeout for remote TTS requests (seconds)",
)

# In speak_text(), add remote mode:
if args.remote:
    config["voice"]["tts"]["mode"] = "remote"
    config["voice"]["tts"]["remote_url"] = args.remote
    config["voice"]["tts"]["remote_timeout"] = args.remote_timeout
```

### 5. Modified File: `main.py` (Startup)

The `_ensure_cuda_dlls()` call and `tts.preload()` need to be mode-aware:

```python
# In run_live_mode(), the CUDA DLL registration should only happen for local TTS:
if tts is not None and tts.mode == "local":
    cuda_dll_found = _ensure_cuda_dlls()

# Pre-load voice models (or check remote server)
if stt is not None:
    # ... existing STT preload ...
if tts is not None:
    tts.preload()  # Works for both local and remote modes
```

## Remote Server Best Practices

### Server Setup (GPU Machine)

1. **Docker deployment** (recommended):
   ```dockerfile
   FROM python:3.13-slim
   WORKDIR /app
   COPY tts_server.py .
   COPY voices/ ./voices/
   RUN pip install piper-tts fastapi uvicorn numpy
   EXPOSE 5000
   CMD ["python", "tts_server.py", "--host", "0.0.0.0", "--port", "5000", "--cuda"]
   ```

2. **Systemd service** (for bare-metal):
   ```ini
   # /etc/systemd/system/piper-tts.service
   [Unit]
   Description=Piper TTS Server
   After=network.target

   [Service]
   Type=simple
   User=piper
   WorkingDirectory=/opt/piper-tts
   ExecStart=/opt/piper-tts/venv/bin/python tts_server.py --host 0.0.0.0 --port 5000 --cuda
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. **Network security**:
   - Bind to `0.0.0.0:5000` on the server
   - Use **SSH tunnel** or **WireGuard VPN** — don't expose raw HTTP to the internet
   - For LAN-only: firewall to only allow the sim rig's IP
   - Optional: add an API key header (`X-API-Key`) for authentication

4. **Performance tuning**:
   - Keep the model loaded in GPU memory (the server does this by default)
   - Use `--workers 1` with uvicorn (single GPU, sequential inference)
   - Expected latency: ~50-200ms for typical race engineer responses on a T4/3060+

### Client Configuration (Sim Rig)

```yaml
voice:
  tts:
    enabled: true
    mode: "remote"
    remote_url: "http://gpu-server:5000"  # Or via SSH tunnel: "http://localhost:5000"
    remote_timeout: 10.0
    # Local settings still available for fallback:
    # model: "en_GB-alan-medium"
    # voice_dir: "voices"
    # use_cuda: false  # CPU fallback
    output_device: null
    volume: 1.0
    sentence_silence: 0.2
```

## Summary of All Changes

| File | Change | Description |
|------|--------|-------------|
| `tts_server.py` | **NEW** | FastAPI server for remote Piper TTS |
| `tts_client.py` | **MODIFY** | Add `mode: "remote"` with HTTP client, keep local mode intact |
| `pyproject.toml` | **MODIFY** | Add `[tts-server]` optional dependency group |
| `test_tts.py` | **MODIFY** | Add `--remote` flag for testing remote server |
| `main.py` | **MODIFY** | Make CUDA DLL registration and preload mode-aware |
| `config.yaml` | **MODIFY** | Add `mode`, `remote_url`, `remote_timeout` to `voice.tts` |

## Key Design Decisions

1. **Dual-mode client** — `mode: "local"` preserves existing behavior exactly; `mode: "remote"` adds the new path. No breaking changes.

2. **WAV response format** — Returns complete WAV files (not raw PCM) so the client can parse sample rate from the header. This is simpler and more robust than negotiating formats.

3. **Volume applied server-side** — The server applies volume so the client receives ready-to-play audio. This keeps the client simple and avoids extra processing on the sim rig.

4. **`requests` library** — Already available as a transitive dependency of `openai`. No new dependency needed on the client side.

5. **Graceful degradation** — If the remote server is unreachable, `speak()` logs an error and returns silently (same as local mode when model fails to load). The spotter audio (WAV files) continues working independently.

6. **SSH tunnel friendly** — The `remote_url` can point to `localhost` when using SSH port forwarding, making it easy to secure the connection without complex TLS setup.
