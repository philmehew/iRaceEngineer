"""
Remote STT client — records audio from the local microphone and transcribes it
via speaches (faster-whisper server) on the Strix Halo server.

Subclasses STTClient to reuse all local recording plumbing (mic device
selection, input gain, push-to-talk). Only transcription is offloaded — the
microphone always runs on this machine via sounddevice.

Config is read from config.yaml under 'voice.stt.remote':
    base_url:   speaches endpoint (e.g. http://192.168.0.117:9000)
    model:      Model name served by speaches (Systran/faster-whisper-small)
    timeout:    HTTP request timeout in seconds
"""

import io
import logging
import wave

import numpy as np

from stt_client import STTClient, WHISPER_SAMPLE_RATE

logger = logging.getLogger(__name__)


def encode_wav(audio: np.ndarray, samplerate: int = WHISPER_SAMPLE_RATE) -> bytes:
    """Encode a float32 mono audio array as a 16-bit WAV file in memory.

    Args:
        audio: Float32 numpy array in [-1, 1].
        samplerate: Sample rate (default 16kHz, expected by Whisper).

    Returns:
        WAV file bytes.
    """
    if len(audio) == 0:
        return b""

    # Convert float32 [-1, 1] to 16-bit PCM
    audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class RemoteSTTClient(STTClient):
    """Record speech locally and transcribe via the speaches server."""

    def __init__(self, config: dict):
        # super() reads shared keys (input_device, input_gain, language) from
        # voice.stt; the local backend keys are read but unused here.
        super().__init__(config)
        remote_config = config.get("voice", {}).get("stt", {}).get("remote", {})
        self.base_url = remote_config.get("base_url", "").rstrip("/")
        self.remote_model = remote_config.get("model", "Systran/faster-whisper-small")
        self.timeout = remote_config.get("timeout", 30.0)
        self._client = None  # Lazy httpx client (shared, thread-safe)

        # Mark the backend so main.py's CUDA-DLL warning (which keys off
        # self.device == "cuda") does not fire for the remote path.
        self.device = "remote"

        logger.info(
            f"Remote STT client configured: {self.base_url} model={self.remote_model}"
        )

    def _get_client(self):
        """Lazily create a shared httpx client for this process."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio via the speaches server.

        Args:
            audio: Float32 numpy array at 16kHz sample rate.

        Returns:
            Transcribed text, or empty string if transcription fails.
        """
        if len(audio) == 0:
            logger.warning("Empty audio — nothing to transcribe")
            return ""

        if not self.base_url:
            logger.error("Remote STT base_url not configured")
            return ""

        wav_bytes = encode_wav(audio, WHISPER_SAMPLE_RATE)
        if not wav_bytes:
            logger.warning("Failed to encode audio to WAV")
            return ""

        logger.info(
            f"Transcribing via remote STT: {len(audio)} samples "
            f"({len(audio) / WHISPER_SAMPLE_RATE:.1f}s)..."
        )

        try:
            response = self._get_client().post(
                f"{self.base_url}/v1/audio/transcriptions",
                data={"model": self.remote_model},
                files={
                    "file": (
                        "audio.wav",
                        wav_bytes,
                        "audio/wav",
                    )
                },
            )
            response.raise_for_status()
            text = response.json().get("text", "").strip()
            logger.info(f'Remote STT transcription: "{text}"')
            return text
        except Exception as e:
            logger.error(f"Remote transcription failed: {e}")
            return ""

    def preload(self):
        """No-op health note — the model is pre-warmed on the server."""
        if not self.base_url:
            logger.warning("Remote STT base_url not configured")
            return
        logger.info(
            f"Remote STT ready — transcription served by {self.base_url} "
            f"(model={self.remote_model})"
        )

    @property
    def is_available(self) -> bool:
        """Check if the remote STT backend is available."""
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return bool(self.base_url)
