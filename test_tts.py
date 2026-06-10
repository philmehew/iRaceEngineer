"""
Standalone TTS test — speak text through Piper TTS.

Tests text-to-speech independently from the main iRaceEngineer pipeline.
Use this to verify Piper setup, voice model, and audio output device.

Usage:
    python test_tts.py "Box this lap, tyres are gone"     # Speak the text
    python test_tts.py --list-devices                      # List output devices
    python test_tts.py --device 5 "Hello"                  # Use output device 5
    python test_tts.py --voice alan "Hello"                   # Use --voice shortcut
    python test_tts.py --model en_GB-alan-medium "Hello"   # Use specific model
    python test_tts.py --download en_GB-alan-medium        # Download a voice model
    python test_tts.py --file response.txt                 # Speak contents of file
    python test_tts.py --wav out.wav "Box this lap"        # Write to WAV (no playback)
"""

import argparse
import sys

import numpy as np

# Short voice name aliases — use with --voice alan, --voice cori, etc.
VOICE_ALIASES: dict[str, str] = {
    "alan": "en_GB-alan-medium",
    "alba": "en_GB-alba-medium",
    "cori": "en_GB-cori-medium",
    "cori-high": "en_GB-cori-high",
    "northern": "en_GB-northern_english_male-medium",
    "northern_english_male": "en_GB-northern_english_male-medium",
}

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


def list_voices(voice_dir: str = "voices"):
    """List available voice models and their aliases."""
    import os

    print("\n🗣️  Voice Aliases (use with --voice):\n")
    for alias, model in VOICE_ALIASES.items():
        onnx = os.path.join(voice_dir, f"{model}.onnx")
        status = "✅ downloaded" if os.path.exists(onnx) else "❌ not downloaded"
        print(f"  {alias:<25} → {model:<40} {status}")

    print("\n📁 All models in voice directory:\n")
    if os.path.isdir(voice_dir):
        models = sorted(f for f in os.listdir(voice_dir) if f.endswith(".onnx"))
        if models:
            for m in models:
                print(f"  {m}")
        else:
            print("  (none found)")
    else:
        print(f"  Directory '{voice_dir}' not found")
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
    wav_path: str | None = None,
):
    """Synthesize and speak text using Piper TTS.

    If wav_path is set, writes audio to a WAV file instead of playing it.
    """
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

    if wav_path:
        audio, sample_rate = tts.synthesize(text)
        if audio.size == 0:
            print("ERROR: Piper produced no audio")
            return
        import wave

        # Convert float32 [-1, 1] to int16 for WAV
        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        print(
            f'✅ Wrote "{text}" to {wav_path} ({sample_rate} Hz, {len(audio) / sample_rate:.1f}s)'
        )
    else:
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
        "--voice",
        metavar="ALIAS",
        help="Voice shortcut (alan, cori, cori-high, northern). Overrides --model.",
    )
    parser.add_argument(
        "--model",
        default="en_GB-alan-medium",
        help="Full Piper voice model name (default: en_GB-alan-medium)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice aliases and downloaded models",
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
        "--wav",
        metavar="PATH",
        help="Write audio to a WAV file instead of playing (useful over SSH)",
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

    if args.list_voices:
        list_voices(args.voice_dir)
        return

    if args.download:
        download_voice(args.download, args.voice_dir)
        return

    # Resolve --voice alias to full model name
    model = args.model
    if args.voice:
        if args.voice in VOICE_ALIASES:
            model = VOICE_ALIASES[args.voice]
        else:
            print(f"ERROR: Unknown voice alias '{args.voice}'")
            print(f"Available: {', '.join(VOICE_ALIASES)}")
            return

    # Get text to speak
    text = ""
    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.text:
        text = " ".join(args.text)

    if not text:
        print(
            "ERROR: Provide text to speak, or use --list-devices / --list-voices / --download"
        )
        parser.print_help()
        return

    speak_text(
        text=text,
        model=model,
        voice_dir=args.voice_dir,
        device=args.device,
        use_cuda=not args.no_cuda,
        volume=args.volume,
        wav_path=args.wav,
    )


if __name__ == "__main__":
    main()
