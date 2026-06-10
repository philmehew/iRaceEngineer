"""
Speech-to-Text client — records audio from microphone and transcribes
using faster-whisper (CTranslate2-based Whisper implementation).

Supports push-to-talk recording, configurable mic device, and GPU/CPU
inference. Model is loaded lazily on first use to avoid slow startup.
"""

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Whisper expects 16kHz mono audio
WHISPER_SAMPLE_RATE = 16000


class STTClient:
    """Record speech from microphone and transcribe using faster-whisper.

    Configuration is read from config.yaml under the 'voice.stt' key:
        model:          Whisper model size (tiny, small, medium, large-v3, turbo)
        device:         Inference device (cuda or cpu)
        compute_type:   Quantization (float16, int8, int8_float16)
        input_device:   Audio input device (null = system default, or name/index)
        vad_filter:     Use Silero VAD to trim silence from recordings
        language:       Language hint for transcription (e.g. "en")
    """

    def __init__(self, config: dict):
        stt_config = config.get("voice", {}).get("stt", {})
        self.model_name = stt_config.get("model", "small")
        self.device = stt_config.get("device", "cuda")
        self.compute_type = stt_config.get("compute_type", "float16")
        self._input_device = stt_config.get("input_device", None)
        self.vad_filter = stt_config.get("vad_filter", True)
        self.language = stt_config.get("language", "en")
        self._model = None
        self._model_loaded = False
        self._load_lock = threading.Lock()

        logger.info(
            f"STT client configured: model={self.model_name}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    def _resolve_input_device(self) -> int | str | None:
        """Resolve input device setting to a sounddevice-compatible value."""
        if self._input_device is None:
            return None
        if isinstance(self._input_device, int):
            return self._input_device
        if isinstance(self._input_device, str):
            try:
                return int(self._input_device)
            except ValueError:
                return self._input_device
        return None

    def _load_model(self):
        """Load the Whisper model (lazy, once). Falls back to CPU if GPU unavailable."""
        if self._model_loaded:
            return

        with self._load_lock:
            if self._model_loaded:
                return

            try:
                from faster_whisper import WhisperModel

                device = self.device
                compute_type = self.compute_type

                # Try CUDA first; fall back to CPU if DLLs or GPU not available
                if device == "cuda":
                    try:
                        logger.info(
                            f"Loading Whisper model: {self.model_name} "
                            f"(device=cuda, compute_type={compute_type})"
                        )
                        self._model = WhisperModel(
                            self.model_name,
                            device="cuda",
                            compute_type=compute_type,
                        )
                    except Exception as gpu_err:
                        logger.warning(
                            f"CUDA unavailable ({gpu_err}), "
                            f"falling back to CPU (int8) — transcription will be slower"
                        )
                        device = "cpu"
                        compute_type = "int8"
                        self._model = WhisperModel(
                            self.model_name,
                            device="cpu",
                            compute_type="int8",
                        )
                else:
                    logger.info(
                        f"Loading Whisper model: {self.model_name} "
                        f"(device={device}, compute_type={compute_type})"
                    )
                    self._model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type,
                    )

                self._model_loaded = True
                logger.info(
                    f"Whisper model loaded: {self.model_name} "
                    f"(device={device}, compute_type={compute_type})"
                )

            except ImportError:
                logger.error(
                    "faster-whisper not installed. Install with: "
                    "uv pip install -e '.[voice]'"
                )
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")

    def record(
        self, duration_s: float, samplerate: int = WHISPER_SAMPLE_RATE
    ) -> np.ndarray:
        """Record audio from the microphone for a fixed duration.

        Args:
            duration_s: Recording duration in seconds.
            samplerate: Sample rate (default 16kHz for Whisper).

        Returns:
            numpy float32 array of recorded audio.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return np.array([], dtype=np.float32)

        device = self._resolve_input_device()
        frames = int(duration_s * samplerate)

        logger.info(f"Recording {duration_s}s from device={device or 'default'}...")
        audio = sd.rec(
            frames,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
        logger.info("Recording complete")

        return audio.flatten()

    def record_until_release(
        self,
        stop_event: threading.Event,
        samplerate: int = WHISPER_SAMPLE_RATE,
        max_duration_s: float = 15.0,
    ) -> np.ndarray:
        """Record audio until stop_event is set (push-to-talk).

        Records in small blocks and concatenates. Stops when:
        - stop_event is set (key released)
        - max_duration_s is reached (safety limit)

        Args:
            stop_event: Threading event — set this to stop recording.
            samplerate: Sample rate (default 16kHz for Whisper).
            max_duration_s: Maximum recording duration in seconds.

        Returns:
            numpy float32 array of recorded audio.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return np.array([], dtype=np.float32)

        device = self._resolve_input_device()
        block_duration = 0.1  # Record in 100ms blocks
        block_frames = int(block_duration * samplerate)
        max_frames = int(max_duration_s * samplerate)

        logger.info("Recording (push-to-talk: release key to stop)...")
        chunks = []
        total_frames = 0

        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=block_frames,
        ) as stream:
            while not stop_event.is_set() and total_frames < max_frames:
                block, overflowed = stream.read(block_frames)
                chunks.append(block.flatten())
                total_frames += len(block)
                if overflowed:
                    logger.debug("Audio buffer overflow — some audio may be choppy")

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        duration = len(audio) / samplerate
        logger.info(f"Recording complete: {duration:.1f}s, {len(audio)} samples")

        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using faster-whisper.

        Args:
            audio: Float32 numpy array at 16kHz sample rate.

        Returns:
            Transcribed text, or empty string if transcription fails.
        """
        if len(audio) == 0:
            logger.warning("Empty audio — nothing to transcribe")
            return ""

        t_start = time.monotonic()

        self._load_model()
        if self._model is None:
            logger.warning("Whisper model not loaded — cannot transcribe")
            return ""

        t_model_loaded = time.monotonic()
        model_load_time = t_model_loaded - t_start
        if model_load_time > 0.1:
            logger.info(f"STT model load: {model_load_time:.3f}s")

        logger.info(
            f"Transcribing {len(audio)} samples "
            f"({len(audio) / WHISPER_SAMPLE_RATE:.1f}s)..."
        )

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self.language,
                vad_filter=self.vad_filter,
                beam_size=2,
            )

            # Force evaluation by consuming the generator
            segment_list = list(segments)
            text = " ".join(s.text.strip() for s in segment_list)

            t_done = time.monotonic()
            logger.info(
                f"STT transcription: {t_done - t_model_loaded:.3f}s "
                f"(total: {t_done - t_start:.3f}s) — "
                f"{info.language}, {info.language_probability:.0%} "
                f'confidence: "{text}"'
            )

            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def listen(self, duration_s: float = 5.0) -> str:
        """Record for a fixed duration and transcribe.

        Convenience method: record + transcribe in one call.

        Args:
            duration_s: Recording duration in seconds.

        Returns:
            Transcribed text.
        """
        audio = self.record(duration_s)
        return self.transcribe(audio)

    def listen_push_to_talk(
        self,
        stop_event: threading.Event,
        max_duration_s: float = 15.0,
    ) -> str:
        """Record until stop_event is set, then transcribe.

        Push-to-talk convenience method.

        Args:
            stop_event: Set this to stop recording.
            max_duration_s: Maximum recording duration.

        Returns:
            Transcribed text.
        """
        audio = self.record_until_release(stop_event, max_duration_s=max_duration_s)
        return self.transcribe(audio)

    @staticmethod
    def list_input_devices() -> list[dict]:
        """List available audio input devices.

        Returns:
            List of dicts with keys: index, name, max_input_channels,
            default_samplerate.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return []

        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append(
                    {
                        "index": i,
                        "name": dev["name"],
                        "max_input_channels": dev["max_input_channels"],
                        "default_samplerate": int(dev["default_samplerate"]),
                    }
                )
        return result

    @property
    def is_available(self) -> bool:
        """Check if STT is available (faster-whisper installed)."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except ImportError:
            return False

    def preload(self):
        """Pre-load the Whisper model to avoid cold-start latency on first use."""
        logger.info("Pre-loading Whisper model...")
        t0 = time.monotonic()
        self._load_model()
        if self._model_loaded:
            logger.info(f"Whisper model pre-loaded in {time.monotonic() - t0:.3f}s")
        else:
            logger.warning("Whisper model pre-load failed")
