"""
Voice client factories — pick the local or remote backend for STT/TTS based on
the 'backend' key in config, so main.py has a single place to make the choice.

Each service reads:
    voice.<stt|tts>.backend: "local" (default) | "remote"

"local" uses the in-process faster-whisper / Piper clients; "remote" uses the
Strix Halo HTTP clients. Both expose the same interface (is_available, preload,
transcribe/speak, etc.), so main.py wiring is unchanged.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stt_client import STTClient
    from tts_client import TTSClient

logger = logging.getLogger(__name__)


def create_stt_client(config: dict) -> "STTClient":
    """Create an STT client for the configured backend (default local)."""
    stt_config = config.get("voice", {}).get("stt", {})
    backend = stt_config.get("backend", "local")

    if backend == "remote":
        from remote_stt_client import RemoteSTTClient

        return RemoteSTTClient(config)

    from stt_client import STTClient

    return STTClient(config)


def create_tts_client(config: dict) -> "TTSClient":
    """Create a TTS client for the configured backend (default local)."""
    tts_config = config.get("voice", {}).get("tts", {})
    backend = tts_config.get("backend", "local")

    if backend == "remote":
        from remote_tts_client import RemoteTTSClient

        return RemoteTTSClient(config)

    from tts_client import TTSClient

    return TTSClient(config)
