"""
Standalone STT test — record from microphone and transcribe with Whisper.

Tests speech-to-text independently from the main iRaceEngineer pipeline.
Use this to verify mic capture, Whisper model setup, and transcription quality.

Usage:
    python test_stt.py                         # Record 5s, transcribe, print
    python test_stt.py --list-devices           # List available input devices
    python test_stt.py --device 3               # Use input device index 3
    python test_stt.py --model small            # Use "small" Whisper model
    python test_stt.py --push-to-talk            # Press Enter to start/stop recording
    python test_stt.py --duration 10            # Record for 10 seconds
"""

import argparse
import sys
import threading

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def list_devices():
    """List all available audio input and output devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: uv pip install -e '.[voice]'")
        return

    devices = sd.query_devices()

    print("\n🎤 Audio Input Devices:\n")
    print(f"{'Idx':<5} {'Name':<50} {'Ch':<5} {'Rate'}")
    print("-" * 70)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " ← default" if i == sd.default.device[0] else ""
            print(
                f"{i:<5} {dev['name']:<50} {dev['max_input_channels']:<5} "
                f"{int(dev['default_samplerate'])}{default}"
            )

    print("\n🔊 Audio Output Devices:\n")
    print(f"{'Idx':<5} {'Name':<50} {'Ch':<5} {'Rate'}")
    print("-" * 70)
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            default = " ← default" if i == sd.default.device[1] else ""
            print(
                f"{i:<5} {dev['name']:<50} {dev['max_output_channels']:<5} "
                f"{int(dev['default_samplerate'])}{default}"
            )
    print()


def test_fixed_duration(
    duration: float,
    model: str = "small",
    device: int | None = None,
    language: str = "en",
    vad: bool = True,
    gain: float = 1.0,
):
    """Record for a fixed duration and transcribe."""
    config = {
        "voice": {
            "stt": {
                "model": model,
                "device": "cpu",
                "compute_type": "int8",
                "input_device": device,
                "input_gain": gain,
                "vad_filter": vad,
                "language": language,
            }
        }
    }

    from stt_client import STTClient

    stt = STTClient(config)

    if not stt.is_available:
        print(
            "ERROR: faster-whisper not available. Install with: uv pip install -e '.[voice]'"
        )
        return

    print(f"🎙️  Recording for {duration}s...")
    print("   Speak now!")
    if device is not None:
        print(f"   Input device: {device}")

    text = stt.listen(duration_s=duration)

    print(f'\n📝 Transcription: "{text}"\n')


def test_push_to_talk(
    model: str = "small",
    device: int | None = None,
    language: str = "en",
    vad: bool = True,
    max_duration: float = 15.0,
    gain: float = 1.0,
):
    """Push-to-talk: press Enter to start recording, press Enter again to stop."""
    config = {
        "voice": {
            "stt": {
                "model": model,
                "device": "cpu",
                "compute_type": "int8",
                "input_device": device,
                "input_gain": gain,
                "vad_filter": vad,
                "language": language,
            }
        }
    }

    from stt_client import STTClient

    stt = STTClient(config)

    if not stt.is_available:
        print(
            "ERROR: faster-whisper not available. Install with: uv pip install -e '.[voice]'"
        )
        return

    stop_event = threading.Event()

    print("🎙️  Push-to-talk mode")
    print("   Press Enter to START recording...")
    input()

    # Start recording in a thread
    result = [""]

    def _record_and_transcribe():
        result[0] = stt.listen_push_to_talk(stop_event, max_duration_s=max_duration)

    rec_thread = threading.Thread(target=_record_and_transcribe)
    rec_thread.start()

    print("   🎤 Recording... Press Enter to STOP.")
    input()
    stop_event.set()

    rec_thread.join(timeout=5)
    print(f'\n📝 Transcription: "{result[0]}"\n')


def main():
    parser = argparse.ArgumentParser(
        description="Standalone STT test — record from microphone and transcribe with Whisper"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input/output devices",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size: tiny, small, medium, large-v3, turbo (default: small)",
    )
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Push-to-talk mode: press Enter to start, Enter again to stop",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection) filter",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for transcription (default: en)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Mic input gain multiplier (default: 1.0, try 2.0 for quiet mics)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum recording duration for push-to-talk (default: 15s)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.push_to_talk:
        test_push_to_talk(
            model=args.model,
            device=args.device,
            language=args.language,
            vad=not args.no_vad,
            max_duration=args.max_duration,
            gain=args.gain,
        )
    else:
        test_fixed_duration(
            duration=args.duration,
            model=args.model,
            device=args.device,
            language=args.language,
            vad=not args.no_vad,
            gain=args.gain,
        )


if __name__ == "__main__":
    main()
