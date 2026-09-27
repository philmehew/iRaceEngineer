"""
Remote TTS client — synthesizes speech via Kokoro-FastAPI on the Strix Halo
server and plays the returned MP3 through the local audio output device.

Subclasses TTSClient to reuse local playback plumbing (output device selection,
volume, playback serialization). Only synthesis is offloaded — playback always
happens on this machine via sounddevice.

Config is read from config.yaml under 'voice.tts.remote':
    base_url:   Kokoro-FastAPI endpoint (e.g. http://192.168.0.117:8880)
    voice:      Voice name (bm_daniel, bm_fable, bm_george, bm_lewis)
    timeout:    HTTP request timeout in seconds
"""

import logging

import numpy as np

from tts_client import TTSClient, preprocess_for_tts

logger = logging.getLogger(__name__)


class RemoteTTSClient(TTSClient):
    """Synthesize speech via Kokoro-FastAPI and play via sounddevice."""

    def __init__(self, config: dict):
        # super() reads shared keys (output_device, volume) from voice.tts;
        # the local backend keys are read but unused by this subclass.
        super().__init__(config)
        remote_config = config.get("voice", {}).get("tts", {}).get("remote", {})
        self.base_url = remote_config.get("base_url", "").rstrip("/")
        self.voice = remote_config.get("voice", "bm_fable")
        self.timeout = remote_config.get("timeout", 30.0)
        self._client = None  # Lazy httpx client (shared, thread-safe)

        logger.info(f"Remote TTS client configured: {self.base_url} voice={self.voice}")

    def _get_client(self):
        """Lazily create a shared httpx client for this process."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _fetch_audio(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to (float32 mono audio, sample_rate) via the server.

        Returns:
            Tuple of (audio_array_float32, sample_rate). On failure returns
            (empty array, 0).
        """
        if not self.base_url:
            logger.error("Remote TTS base_url not configured")
            return np.array([], dtype=np.float32), 0

        try:
            response = self._get_client().post(
                f"{self.base_url}/v1/audio/speech",
                json={"input": text, "voice": self.voice},
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Remote TTS synthesis failed: {e}")
            return np.array([], dtype=np.float32), 0

        try:
            import miniaudio

            # Kokoro-FastAPI returns MP3 audio bytes.
            decoded = miniaudio.decode(
                response.content, output_format=miniaudio.SampleFormat.FLOAT32
            )
            audio = np.frombuffer(decoded.samples, dtype=np.float32)
            if decoded.nchannels > 1:
                audio = audio.reshape(-1, decoded.nchannels).mean(axis=1)
            return audio, decoded.sample_rate
        except ImportError:
            logger.error(
                "miniaudio not installed — cannot decode remote TTS audio. "
                "Install with: uv sync"
            )
            return np.array([], dtype=np.float32), 0
        except Exception as e:
            logger.error(f"Remote TTS audio decode failed: {e}")
            return np.array([], dtype=np.float32), 0

    def speak(self, text: str):
        """Synthesize text on the server and play through the local device.

        Thread-safe: playback is serialized via the inherited _play_lock.
        """
        if not text.strip():
            return

        text = preprocess_for_tts(text)

        try:
            import sounddevice as sd
        except ImportError:
            logger.error(
                "sounddevice not installed. Install with: uv sync --extra voice"
            )
            return

        device = self._resolve_output_device()
        logger.info(f"Speaking (remote): {text[:80]}{'...' if len(text) > 80 else ''}")

        with self._play_lock:
            try:
                audio, sample_rate = self._fetch_audio(text)
                if audio.size == 0 or sample_rate == 0:
                    logger.warning("Remote TTS produced no audio")
                    return

                if self.volume != 1.0:
                    audio = np.clip(audio * self.volume, -1.0, 1.0)

                sd.play(audio, samplerate=sample_rate, device=device)
                sd.wait()
            except Exception as e:
                logger.error(f"TTS playback failed: {e}")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to a numpy array without playing.

        Returns:
            Tuple of (audio_array_float32, sample_rate).
        """
        audio, sample_rate = self._fetch_audio(text)
        if self.volume != 1.0 and audio.size > 0:
            audio = np.clip(audio * self.volume, -1.0, 1.0)
        return audio, sample_rate

    def preload(self):
        """Health-check the server and confirm the configured voice exists.

        Non-fatal: logs a warning if the server is unreachable so the
        application still starts (mirrors local TTS lazy-load behaviour).
        """
        if not self.base_url:
            logger.warning("Remote TTS base_url not configured")
            return

        try:
            response = self._get_client().get(f"{self.base_url}/v1/audio/voices")
            response.raise_for_status()
            payload = response.json()
            # Kokoro-FastAPI returns {"voices": [{"id": ..., "name": ...}, ...]}
            voice_list = (
                payload.get("voices", payload) if isinstance(payload, dict) else payload
            )
            names = [v.get("name") or v.get("id") for v in voice_list]
            if names and self.voice not in names:
                logger.warning(
                    f"Remote TTS voice '{self.voice}' not in catalogue: {names}"
                )
            else:
                logger.info(f"Remote TTS server reachable, voice '{self.voice}' OK")
        except Exception as e:
            logger.warning(f"Remote TTS server unreachable: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the remote TTS backend is available."""
        try:
            import httpx  # noqa: F401
            import miniaudio  # noqa: F401
        except ImportError:
            return False
        return bool(self.base_url)
