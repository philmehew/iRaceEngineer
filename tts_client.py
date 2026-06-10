"""
Text-to-Speech client — synthesizes text using Piper TTS and plays audio
through a selected output device via sounddevice.

Supports configurable voice models, GPU acceleration, and audio output
device selection. Model is loaded lazily on first use to avoid slow startup.
"""

import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class TTSClient:
    """Synthesize text to speech using Piper TTS and play via sounddevice.

    Configuration is read from config.yaml under the 'voice.tts' key:
        model:          Piper voice model name (e.g. en_GB-alan-medium)
        voice_dir:      Directory where voice .onnx files are stored
        use_cuda:       Enable GPU acceleration for Piper
        output_device:  Audio output device (null = system default, or name/index)
        volume:         Volume multiplier (1.0 = normal)
        sentence_silence: Seconds of silence between sentences
    """

    def __init__(self, config: dict):
        tts_config = config.get("voice", {}).get("tts", {})
        self.model_name = tts_config.get("model", "en_GB-alan-medium")
        self.voice_dir = tts_config.get("voice_dir", "voices")
        self.use_cuda = tts_config.get("use_cuda", True)
        self.volume = tts_config.get("volume", 1.0)
        self.sentence_silence = tts_config.get("sentence_silence", 0.2)
        self._output_device = tts_config.get("output_device", None)
        self._voice = None
        self._voice_loaded = False
        self._load_lock = threading.Lock()

        logger.info(
            f"TTS client configured: model={self.model_name}, "
            f"voice_dir={self.voice_dir}, cuda={self.use_cuda}"
        )

    def _resolve_output_device(self) -> int | str | None:
        """Resolve output device setting to a sounddevice-compatible value."""
        if self._output_device is None:
            return None
        if isinstance(self._output_device, int):
            return self._output_device
        if isinstance(self._output_device, str):
            try:
                return int(self._output_device)
            except ValueError:
                return self._output_device
        return None

    def _load_voice(self):
        """Load the Piper voice model (lazy, once)."""
        if self._voice_loaded:
            return

        with self._load_lock:
            if self._voice_loaded:
                return

            try:
                from piper import PiperVoice

                model_path = self._find_model_path()
                if not model_path:
                    logger.error(
                        f"Piper model not found: {self.model_name}. "
                        f"Run: python -m piper.download_voices {self.model_name} "
                        f"--data-dir {self.voice_dir}"
                    )
                    return

                logger.info(f"Loading Piper voice model: {model_path}")
                self._voice = PiperVoice.load(model_path, use_cuda=self.use_cuda)
                self._voice_loaded = True
                logger.info(f"Piper voice loaded: {self.model_name}")

            except ImportError:
                logger.error(
                    "piper-tts not installed. Install with: "
                    "uv pip install -e '.[voice]'"
                )
            except Exception as e:
                logger.error(f"Failed to load Piper voice: {e}")

    def _find_model_path(self) -> str | None:
        """Find the .onnx model file for the configured voice."""
        # Check voice_dir for model_name.onnx
        model_file = os.path.join(self.voice_dir, f"{self.model_name}.onnx")
        if os.path.exists(model_file):
            return model_file

        # Check voice_dir/model_name/ directory
        model_dir = os.path.join(self.voice_dir, self.model_name)
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".onnx"):
                    return os.path.join(model_dir, f)

        # Check current directory (piper downloads here by default)
        model_file = f"{self.model_name}.onnx"
        if os.path.exists(model_file):
            return model_file

        return None

    def speak(self, text: str):
        """Synthesize text and play through the configured audio device.

        Streams audio sentence-by-sentence for lower latency.
        """
        if not text.strip():
            return

        self._load_voice()
        if self._voice is None:
            logger.warning("TTS voice not loaded — skipping speech")
            return

        try:
            import sounddevice as sd
        except ImportError:
            logger.error(
                "sounddevice not installed. Install with: uv pip install -e '.[voice]'"
            )
            return

        device = self._resolve_output_device()
        logger.info(f"Speaking: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            t_start = time.monotonic()

            # Collect audio chunks from streaming synthesis
            audio_chunks = []
            sample_rate = None

            for chunk in self._voice.synthesize(text.strip()):
                if sample_rate is None:
                    sample_rate = chunk.sample_rate
                audio_chunks.append(
                    np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                )

            if not audio_chunks or sample_rate is None:
                logger.warning("Piper produced no audio")
                return

            t_synthesized = time.monotonic()
            logger.info(
                f"TTS synthesis: {t_synthesized - t_start:.3f}s "
                f"({len(audio_chunks)} chunks)"
            )

            audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0

            # Apply volume
            if self.volume != 1.0:
                audio = audio * self.volume

            # Play through selected device
            sd.play(audio, samplerate=sample_rate, device=device)
            sd.wait()

            t_done = time.monotonic()
            logger.info(
                f"TTS playback: {t_done - t_synthesized:.3f}s, "
                f"total speak: {t_done - t_start:.3f}s"
            )

        except Exception as e:
            logger.error(f"TTS playback failed: {e}")

    def speak_async(self, text: str):
        """Synthesize and play audio in a background thread.

        Non-blocking — returns immediately while audio plays.
        """
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to a numpy array without playing.

        Returns:
            Tuple of (audio_array_float32, sample_rate).
        """
        self._load_voice()
        if self._voice is None:
            return np.array([], dtype=np.float32), 22050

        audio_chunks = []
        sample_rate = None

        for chunk in self._voice.synthesize(text.strip()):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
            audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

        if not audio_chunks or sample_rate is None:
            return np.array([], dtype=np.float32), 22050

        audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0

        if self.volume != 1.0:
            audio = audio * self.volume

        return audio, sample_rate

    @staticmethod
    def list_output_devices() -> list[dict]:
        """List available audio output devices.

        Returns:
            List of dicts with keys: index, name, max_output_channels,
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
            if dev["max_output_channels"] > 0:
                result.append(
                    {
                        "index": i,
                        "name": dev["name"],
                        "max_output_channels": dev["max_output_channels"],
                        "default_samplerate": int(dev["default_samplerate"]),
                    }
                )
        return result

    @property
    def is_available(self) -> bool:
        """Check if TTS is available (model loaded or loadable)."""
        try:
            from piper import PiperVoice  # noqa: F401

            return True
        except ImportError:
            return False

    def preload(self):
        """Pre-load the Piper voice model to avoid cold-start latency on first use."""
        logger.info("Pre-loading Piper voice model...")
        t0 = time.monotonic()
        self._load_voice()
        if self._voice_loaded:
            logger.info(f"Piper voice pre-loaded in {time.monotonic() - t0:.3f}s")
        else:
            logger.warning("Piper voice pre-load failed")
