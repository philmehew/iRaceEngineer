"""Tests for remote voice clients and backend selection (offline, no network)."""

import numpy as np

from remote_stt_client import encode_wav


class TestEncodeWav:
    def test_valid_wav_header(self):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        wav_bytes = encode_wav(audio, samplerate=16000)
        assert wav_bytes.startswith(b"RIFF")
        assert wav_bytes[8:12] == b"WAVE"

    def test_sample_count_and_rate(self):
        import io
        import wave

        audio = np.array([0.0, 0.1, -0.1, 0.2], dtype=np.float32)
        wav_bytes = encode_wav(audio, samplerate=16000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 4

    def test_round_trip(self):
        # int16 quantisation means the round-trip is exact for these values
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav_bytes = encode_wav(audio, samplerate=16000)

        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        expected = (np.clip(audio * 32768, -32768, 32767)).astype(np.int16)
        assert np.array_equal(frames, expected)

    def test_empty_audio(self):
        assert encode_wav(np.array([], dtype=np.float32)) == b""


class TestVoiceClients:
    def test_default_backend_is_local(self):
        from voice_clients import create_stt_client, create_tts_client
        from stt_client import STTClient
        from tts_client import TTSClient

        config = {"voice": {"stt": {}, "tts": {}}}
        assert isinstance(create_stt_client(config), STTClient)
        assert isinstance(create_tts_client(config), TTSClient)

    def test_remote_backend_selected(self):
        from voice_clients import create_stt_client, create_tts_client
        from remote_stt_client import RemoteSTTClient
        from remote_tts_client import RemoteTTSClient

        config = {
            "voice": {
                "stt": {
                    "backend": "remote",
                    "remote": {"base_url": "http://example.invalid:9000"},
                },
                "tts": {
                    "backend": "remote",
                    "remote": {"base_url": "http://example.invalid:8880"},
                },
            }
        }
        assert isinstance(create_stt_client(config), RemoteSTTClient)
        assert isinstance(create_tts_client(config), RemoteTTSClient)

    def test_local_client_reads_nested_config(self):
        """Local clients read backend keys from 'local:' when present."""
        from voice_clients import create_stt_client, create_tts_client

        config = {
            "voice": {
                "stt": {
                    "backend": "local",
                    "local": {"model": "tiny", "device": "cpu"},
                },
                "tts": {
                    "backend": "local",
                    "local": {"model": "en_GB-alan-medium", "use_cuda": False},
                },
            }
        }
        stt = create_stt_client(config)
        tts = create_tts_client(config)
        assert stt.model_name == "tiny"
        assert stt.device == "cpu"
        assert tts.model_name == "en_GB-alan-medium"
        assert tts.use_cuda is False

    def test_local_client_falls_back_to_flat_config(self):
        """Backward compat: flat (non-nested) config still works."""
        from voice_clients import create_stt_client, create_tts_client

        config = {
            "voice": {
                "stt": {"model": "medium", "device": "cpu"},
                "tts": {"model": "en_GB-cori-medium"},
            }
        }
        stt = create_stt_client(config)
        tts = create_tts_client(config)
        assert stt.model_name == "medium"
        assert stt.device == "cpu"
        assert tts.model_name == "en_GB-cori-medium"
