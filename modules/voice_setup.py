"""
Voice client setup — graceful TTS/STT construction shared by live and replay modes.

Wraps the voice_clients factories with the degrade-gracefully-if-unavailable
logic that used to be inline in both run modes, plus CUDA DLL registration and
model preloading to avoid cold-start latency on the first PTT press.
"""

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.stt_client import STTClient
    from modules.tts_client import TTSClient

logger = logging.getLogger("iraceengineer")


def _ensure_cuda_dlls() -> bool:
    """Register CUDA toolkit DLL directory on Windows.

    The CUDA installer sometimes doesn't add itself to PATH, causing
    "DLL not found" errors from ctranslate2/faster-whisper. This scans
    for the toolkit and registers the bin directory so the DLLs are findable.

    Must be called before any code that loads CUDA (e.g. stt_client preload).

    Returns:
        True if CUDA DLL directory was found and registered, False otherwise.
    """
    if sys.platform != "win32":
        return False

    # Try CUDA_PATH env var first (set by some installer versions)
    cuda_base = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
    if os.path.isdir(cuda_base):
        os.add_dll_directory(cuda_base)
        return True

    # Fallback: scan the standard toolkit install location
    toolkit_dir = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        "NVIDIA GPU Computing Toolkit",
        "CUDA",
    )
    if not os.path.isdir(toolkit_dir):
        return False

    # Pick the latest installed version
    versions = sorted(
        (d for d in os.listdir(toolkit_dir) if d.startswith("v")),
        reverse=True,
    )
    if not versions:
        return False

    cuda_bin = os.path.join(toolkit_dir, versions[0], "bin")
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
        return True

    return False


def setup_tts(config: dict) -> "TTSClient | None":
    """Gracefully construct a TTS client — None if disabled or unavailable.

    Catches construction failures and backend-unavailable states so live and
    replay modes share the same degrade-gracefully behaviour.
    """
    voice_config = config.get("voice", {})
    if not voice_config.get("tts", {}).get("enabled", False):
        return None

    from modules.voice_clients import create_tts_client

    try:
        tts = create_tts_client(config)
        if not tts.is_available:
            logger.warning("TTS configured but backend unavailable — disabled")
            return None
        logger.info("TTS enabled — LLM responses will be spoken aloud")
        return tts
    except Exception as e:
        logger.warning(f"TTS setup failed: {e}")
        return None


def setup_stt(config: dict) -> "STTClient | None":
    """Gracefully construct an STT client — None if disabled or unavailable.

    Catches construction failures and backend-unavailable states so live and
    replay modes share the same degrade-gracefully behaviour.
    """
    voice_config = config.get("voice", {})
    if not voice_config.get("stt", {}).get("enabled", False):
        return None

    from modules.voice_clients import create_stt_client

    try:
        stt = create_stt_client(config)
        if not stt.is_available:
            logger.warning("STT configured but backend unavailable — disabled")
            return None
        logger.info("STT enabled — voice input available via push-to-talk")
        return stt
    except Exception as e:
        logger.warning(f"STT setup failed: {e}")
        return None


def preload_voice_models(stt=None, tts=None) -> None:
    """Register CUDA DLLs and pre-load voice models to avoid cold-start latency.

    Must run before any code that loads CUDA (e.g. stt_client preload).
    Warns when STT wants CUDA but the toolkit DLLs can't be found (falls back
    to CPU).
    """
    cuda_dll_found = _ensure_cuda_dlls()

    if stt is not None:
        if getattr(stt, "device", None) == "cuda" and not cuda_dll_found:
            logger.warning(
                "CUDA toolkit DLLs not found — STT will fall back to CPU. "
                "Install CUDA or add it to PATH for GPU acceleration."
            )
        stt.preload()
    if tts is not None:
        tts.preload()
