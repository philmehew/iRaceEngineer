"""
Standalone TTS test — speak text through Piper TTS.

Tests text-to-speech independently from the main iRaceEngineer pipeline.
Use this to verify Piper setup, voice model, and audio output device.

Usage:
    python test_tts.py "Box this lap, tyres are gone"     # Speak the text
    python test_tts.py --list-devices                      # List output devices
    python test_tts.py --device 5 "Hello"                  # Use output device 5
    python test_tts.py --model en_GB-alan-medium "Hello"   # Use specific model
    python test_tts.py --download en_GB-alan-medium        # Download a voice model
    python test_tts.py --file response.txt                 # Speak contents of file
"""

import argparse
import sys

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def list_devices():
    """List all available audio output devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: uv pip install -e '.[voice]'")
        return

    devices = sd.query_devices()
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

    # Also show input devices for reference
    print("🎤 Audio Input Devices:\n")
    print(f"{'Idx':<5} {'Name':<50} {'Ch':<5} {'Rate'}")
    print("-" * 70)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " ← default" if i == sd.default.device[0] else ""
            print(
                f"{i:<5} {dev['name']:<50} {dev['max_input_channels']:<5} "
                f"{int(dev['default_samplerate'])}{default}"
            )
    print()


def download_voice(model_name: str, voice_dir: str):
    """Download a Piper voice model."""
    print(f"Downloading Piper voice model: {model_name}")
    print(f"Voice directory: {voice_dir}")

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            model_name,
            "--data-dir",
            voice_dir,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Download failed:\n{result.stderr}")
        return

    print(result.stdout)
    print(f"✅ Voice model downloaded to {voice_dir}")


def speak_text(
    text: str,
    model: str = "en_GB-alan-medium",
    voice_dir: str = "voices",
    device: int | None = None,
    use_cuda: bool = True,
    volume: float = 1.0,
):
    """Synthesize and speak text using Piper TTS."""
    config = {
        "voice": {
            "tts": {
                "model": model,
                "voice_dir": voice_dir,
                "use_cuda": use_cuda,
                "output_device": device,
                "volume": volume,
            }
        }
    }

    from tts_client import TTSClient

    tts = TTSClient(config)

    if not tts.is_available:
        print(
            "ERROR: Piper TTS not available. Install with: uv pip install -e '.[voice]'"
        )
        return

    print(f'🗣️  Speaking: "{text}"')
    print(f"   Model: {model}")
    if device is not None:
        print(f"   Device: {device}")

    tts.speak(text)
    print("✅ Done")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone TTS test — speak text through Piper TTS"
    )
    parser.add_argument("text", nargs="*", help="Text to speak")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio output devices",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio output device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--model",
        default="en_GB-alan-medium",
        help="Piper voice model name (default: en_GB-alan-medium)",
    )
    parser.add_argument(
        "--voice-dir",
        default="voices",
        help="Directory where voice .onnx files are stored (default: voices)",
    )
    parser.add_argument(
        "--download",
        metavar="MODEL",
        help="Download a Piper voice model (e.g. en_GB-alan-medium)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Read text to speak from a file",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Volume multiplier (default: 1.0)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.download:
        download_voice(args.download, args.voice_dir)
        return

    # Get text to speak
    text = ""
    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.text:
        text = " ".join(args.text)

    if not text:
        print("ERROR: Provide text to speak, or use --list-devices / --download")
        parser.print_help()
        return

    speak_text(
        text=text,
        model=args.model,
        voice_dir=args.voice_dir,
        device=args.device,
        use_cuda=not args.no_cuda,
        volume=args.volume,
    )


if __name__ == "__main__":
    main()
