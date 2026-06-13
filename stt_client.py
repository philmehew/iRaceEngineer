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
        input_gain:     Mic input gain multiplier (1.0 = no boost, 2.0 = 2x louder)
        vad_filter:     Use Silero VAD to trim silence from recordings
        language:       Language hint for transcription (e.g. "en")
    """

    def __init__(self, config: dict):
        stt_config = config.get("voice", {}).get("stt", {})
        self.model_name = stt_config.get("model", "small")
        self.device = stt_config.get("device", "cuda")
        self.compute_type = stt_config.get("compute_type", "float16")
        self._input_device = stt_config.get("input_device", None)
        self.input_gain = stt_config.get("input_gain", 1.0)
        self.vad_filter = stt_config.get("vad_filter", True)
        self.language = stt_config.get("language", "en")
        self._model = None
        self._model_loaded = False
        self._cpu_model = None  # Lazy-loaded CPU fallback for CUDA failures
        self._cpu_model_loaded = False
        self._cpu_load_lock = threading.Lock()
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

    def _load_cpu_model(self):
        """Load a CPU fallback Whisper model for when CUDA fails mid-race.

        Lazily loaded on first CUDA failure. Uses int8 quantization for
        minimal CPU overhead. Slower than CUDA (~2-5x) but won't compete
        with iRacing for GPU resources or crash on CUDA OOM.
        """
        if self._cpu_model_loaded:
            return

        with self._cpu_load_lock:
            if self._cpu_model_loaded:
                return

            try:
                from faster_whisper import WhisperModel

                logger.info(
                    f"Loading CPU fallback Whisper model: {self.model_name} "
                    f"(device=cpu, compute_type=int8)"
                )
                self._cpu_model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                )
                self._cpu_model_loaded = True
                logger.info(f"CPU fallback Whisper model loaded: {self.model_name}")

            except ImportError:
                logger.error("faster-whisper not installed — cannot load CPU fallback")
            except Exception as e:
                logger.error(f"Failed to load CPU fallback Whisper model: {e}")

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

        audio = audio.flatten()

        # Apply input gain
        if self.input_gain != 1.0:
            audio = audio * self.input_gain

        return audio

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

        If the audio device can't be opened (e.g. grabbed by another app),
        waits up to 3 seconds for it to become available before giving up.

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

        # Try to open the input stream — retry for up to 3s if the device
        # is busy (e.g. iRacing or another app has it). This prevents the
        # recording thread from hanging forever on a blocked device.
        stream = None
        open_deadline = time.monotonic() + 3.0
        last_error = None
        retries = 0
        while time.monotonic() < open_deadline:
            if stop_event.is_set():
                logger.warning("PTT released before recording started")
                return np.array([], dtype=np.float32)
            try:
                stream = sd.InputStream(
                    samplerate=samplerate,
                    channels=1,
                    dtype="float32",
                    device=device,
                    blocksize=block_frames,
                )
                stream.start()
                if retries > 0:
                    logger.warning(
                        f"Audio input device opened after {retries} retry/ies"
                    )
                break
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"Audio input device busy, retrying... ({e})")
                time.sleep(0.3)

        if stream is None:
            logger.error(
                f"Could not open audio input device after 3s — "
                f"another app may be using it: {last_error}"
            )
            return np.array([], dtype=np.float32)

        try:
            while not stop_event.is_set() and total_frames < max_frames:
                block, overflowed = stream.read(block_frames)
                chunks.append(block.flatten())
                total_frames += len(block)
                if overflowed:
                    logger.debug("Audio buffer overflow — some audio may be choppy")
        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        # Apply input gain
        if self.input_gain != 1.0:
            audio = audio * self.input_gain

        duration = len(audio) / samplerate
        logger.info(f"Recording complete: {duration:.1f}s, {len(audio)} samples")

        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using faster-whisper.

        If CUDA transcription fails (e.g. GPU OOM when iRacing is using VRAM),
        automatically falls back to CPU inference so the PTT pipeline doesn't
        die mid-race.

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

        # If on CUDA, run transcription in a worker thread with a timeout.
        # When iRacing is contending for GPU, CUDA inference can slow to a crawl
        # (49s+ observed) without actually throwing an error — the generator
        # just yields segments very slowly. A timeout lets us abort the
        # starving CUDA attempt and fall through to CPU fallback.
        cuda_timeout_s = 10.0 if self.device == "cuda" else None

        try:
            text = self._do_transcribe(
                self._model, audio, t_start, t_model_loaded, timeout_s=cuda_timeout_s
            )
            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")

            # Auto-fallback: if we're on CUDA, retry on CPU. iRacing and
            # Whisper competing for GPU can cause CUDA OOM/timeout errors
            # mid-race. CPU is slower (~2-5x) but won't crash or stutter
            # the game, and a working slow response beats a broken fast one.
            if self.device == "cuda" and self._model is not None:
                logger.info(
                    "Retrying transcription on CPU (CUDA may be contested by iRacing)..."
                )
                try:
                    self._load_cpu_model()
                    if self._cpu_model is not None:
                        text = self._do_transcribe(
                            self._cpu_model,
                            audio,
                            t_start,
                            t_model_loaded,
                            label="CPU fallback",
                        )
                        return text
                except Exception as cpu_err:
                    logger.error(f"CPU fallback transcription also failed: {cpu_err}")

            return ""

    def _do_transcribe(
        self,
        model,
        audio: np.ndarray,
        t_start: float,
        t_model_loaded: float,
        timeout_s: float | None = None,
        label: str = "STT",
    ) -> str:
        """Run transcription on a model, with optional timeout.

        Args:
            model: WhisperModel instance to use.
            audio: Float32 numpy array at 16kHz.
            t_start: Monotonic time when transcribe() was entered.
            t_model_loaded: Monotonic time when model was confirmed loaded.
            timeout_s: Abort transcription if it takes longer than this.
                None = no timeout. Set for CUDA to avoid GPU contention hangs.
            label: Label for log messages (e.g. "CPU fallback").

        Returns:
            Transcribed text.

        Raises:
            TimeoutError: If transcription exceeds timeout_s.
            Exception: If transcription fails for any other reason.
        """
        audio_duration = len(audio) / WHISPER_SAMPLE_RATE

        if timeout_s is not None:
            # Run transcription in a worker thread so we can enforce a timeout.
            # faster-whisper's transcribe() returns a lazy generator — the
            # actual inference happens when we consume it via list().
            result_box: list = [None]  # [text] on success
            error_box: list = [None]  # [exception] on failure

            def _worker():
                try:
                    segments, info = model.transcribe(
                        audio,
                        language=self.language,
                        vad_filter=self.vad_filter,
                        beam_size=2,
                    )
                    segment_list = list(segments)
                    result_box[0] = " ".join(s.text.strip() for s in segment_list)
                except Exception as e:
                    error_box[0] = e

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()
            worker.join(timeout=timeout_s)

            if worker.is_alive():
                # Thread didn't finish in time — CUDA is likely starved
                logger.warning(
                    f"{label} transcription timed out after {timeout_s:.0f}s "
                    f"({audio_duration:.1f}s audio) — GPU likely contested by iRacing"
                )
                raise TimeoutError(f"Transcription timed out after {timeout_s:.0f}s")

            if error_box[0] is not None:
                raise error_box[0]

            text = result_box[0] or ""
        else:
            # No timeout — run directly (CPU mode, no contention risk)
            segments, info = model.transcribe(
                audio,
                language=self.language,
                vad_filter=self.vad_filter,
                beam_size=2,
            )
            segment_list = list(segments)
            text = " ".join(s.text.strip() for s in segment_list)

        t_done = time.monotonic()
        logger.info(
            f"{label} transcription: {t_done - t_model_loaded:.3f}s "
            f"(total: {t_done - t_start:.3f}s) — "
            f'confidence: "{text}"'
        )

        return text

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
