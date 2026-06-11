"""
Spotter module — deterministic real-time audio calls for car proximity.

Reads CarLeftRight from iRacing telemetry, detects transitions (car appears/
disappears alongside), and plays pre-recorded WAV files with low latency.

Architecture:
    iRacing telemetry (30Hz)
           │
           ▼
    ProximityDetector   ← edge-detect CarLeftRight transitions, enforce cooldowns
           │
           ▼
         Spotter           ← coordinator: owns detector + audio player, called each tick
           │
           ▼
    SpotterAudioPlayer   ← loads WAVs into memory at startup, plays via sounddevice.OutputStream

Key design decisions:
- Uses sounddevice.OutputStream (not sd.play) to avoid killing TTS playback.
- Edge detection, not level detection — fires only on transitions, not every tick.
- Cooldown timers prevent repeated calls within a configurable window.
- Guard: only processes CarLeftRight when the player is on track.
"""

import logging
import os
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpotterCall:
    """A single spotter call event to be played as audio."""

    call_type: str  # e.g. "car_left", "car_right", "three_wide", "clear"
    priority: int  # 0=safety_critical, 1=team_wide, 2=driver_specific, 3=on_demand
    audio_key: str  # key into the pre-loaded audio samples dict


class ProximityDetector:
    """State machine that detects CarLeftRight transitions.

    iRacing CarLeftRight values:
        0 = no car alongside
        1 = car on the left
        2 = car on the right
        3 = car on both sides

    The detector tracks the previous state per direction and fires events
    only on transitions (rising edges for appearance, falling edges for
    clearance). A cooldown timer prevents repeated calls within a
    configurable window.

    Special case: when transitioning from 0→3 (both sides at once from
    none), a single "three_wide" call is emitted instead of separate
    car_left + car_right calls.
    """

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        cooldowns = spotter_config.get("cooldowns", {})
        self._proximity_cooldown = cooldowns.get("proximity_ms", 3000) / 1000.0
        self._clearance_cooldown = cooldowns.get("clearance_ms", 5000) / 1000.0
        self._clear_delay = cooldowns.get("clear_delay_ms", 500) / 1000.0

        self._prev_left = False
        self._prev_right = False
        self._last_call_time: dict[str, float] = {}

        # Whether a car has been alongside since the last "clear" call.
        # Prevents "clear" from firing before any proximity call was made
        # (e.g. at session start) or while a car is still alongside.
        self._car_alongside: bool = False

        # Pending clear tracking — debounce to avoid false "clear" calls
        # from telemetry flicker (car momentarily reads as gone).
        # Set to the monotonic time when the side first became clear;
        # None means the side is occupied or no pending clear.
        self._pending_clear_since_left: float | None = None
        self._pending_clear_since_right: float | None = None

    def update(self, car_left_right: int, current_time: float) -> list[SpotterCall]:
        """Process a CarLeftRight value and return any calls to play.

        Args:
            car_left_right: iRacing CarLeftRight value (0=none, 1=left, 2=right, 3=both)
            current_time: Monotonic time for cooldown tracking (time.monotonic())

        Returns:
            List of SpotterCall events to play.
        """
        cur_left = bool(car_left_right & 1)
        cur_right = bool(car_left_right & 2)

        calls: list[SpotterCall] = []

        # --- Debounced clear detection ---
        # Instead of firing "clear" the instant telemetry reads the car as gone,
        # we require the car to be gone for clear_delay seconds. This prevents
        # false "clear" calls from telemetry flicker (1-2 frames of 0 when the
        # car is actually still alongside).

        if cur_left:
            # Car is present on left — cancel any pending clear
            self._pending_clear_since_left = None
        elif self._prev_left:
            # Car just disappeared from left — start pending clear timer
            self._pending_clear_since_left = current_time

        if cur_right:
            # Car is present on right — cancel any pending clear
            self._pending_clear_since_right = None
        elif self._prev_right:
            # Car just disappeared from right — start pending clear timer
            self._pending_clear_since_right = current_time

        # Check if any pending clears have matured past the delay threshold
        left_clear_matured = (
            self._pending_clear_since_left is not None
            and (current_time - self._pending_clear_since_left) >= self._clear_delay
        )
        right_clear_matured = (
            self._pending_clear_since_right is not None
            and (current_time - self._pending_clear_since_right) >= self._clear_delay
        )

        # --- Appearance detection ---
        # Detect transitions. A side "appears" if it was NOT present last tick
        # AND there is no pending clear for that side (meaning the car truly
        # left and came back, not just a flicker).
        left_appeared = cur_left and not self._prev_left
        right_appeared = cur_right and not self._prev_right

        # If a car reappears while a clear was pending (flicker, not a real
        # departure), cancel the pending clear and suppress the re-appearance
        # call — the car never actually left.
        if cur_left and self._pending_clear_since_left is not None:
            # Car came back before "clear" was announced — it was a flicker.
            # Cancel the pending clear and treat this as if the car never left.
            self._pending_clear_since_left = None
            left_appeared = False  # Don't announce re-appearance for a flicker

        if cur_right and self._pending_clear_since_right is not None:
            self._pending_clear_since_right = None
            right_appeared = False

        # Appearance calls
        if left_appeared or right_appeared:
            # Special case: both sides appeared simultaneously from none → three_wide
            if (
                left_appeared
                and right_appeared
                and not self._prev_left
                and not self._prev_right
            ):
                if self._cooldown_elapsed("three_wide", current_time):
                    calls.append(
                        SpotterCall(
                            call_type="three_wide",
                            priority=0,
                            audio_key="three_wide",
                        )
                    )
                    self._last_call_time["three_wide"] = current_time
                    self._car_alongside = True
                    # Reset proximity cooldowns for individual sides too, so
                    # a rapid 0→3→0→1 sequence doesn't re-trigger left immediately
                    self._last_call_time["car_left"] = current_time
                    self._last_call_time["car_right"] = current_time
            else:
                # Individual side appearance
                if left_appeared:
                    if self._cooldown_elapsed("car_left", current_time):
                        calls.append(
                            SpotterCall(
                                call_type="car_left",
                                priority=0,
                                audio_key="car_left",
                            )
                        )
                        self._last_call_time["car_left"] = current_time
                        self._car_alongside = True

                if right_appeared:
                    if self._cooldown_elapsed("car_right", current_time):
                        calls.append(
                            SpotterCall(
                                call_type="car_right",
                                priority=0,
                                audio_key="car_right",
                            )
                        )
                        self._last_call_time["car_right"] = current_time
                        self._car_alongside = True

        # Clearance calls — only after the clear delay has elapsed.
        # Additional guards:
        #   1. A car must have been alongside since the last "clear" call
        #      (_car_alongside) — prevents "clear" at session start or
        #      repeating "clear" when no car was ever there.
        #   2. No car can be alongside right now (cur_left/cur_right) —
        #      prevents "clear" playing simultaneously with a proximity call
        #      (e.g. car leaves one side but appears on the other).
        if left_clear_matured or right_clear_matured:
            if (
                self._car_alongside
                and not cur_left
                and not cur_right
                and self._cooldown_elapsed("clear", current_time)
            ):
                calls.append(
                    SpotterCall(
                        call_type="clear",
                        priority=0,
                        audio_key="clear",
                    )
                )
                self._last_call_time["clear"] = current_time
                self._car_alongside = False
            # Clear the pending timers regardless — we only fire once.
            # If we suppressed "clear" because a car is alongside on the
            # other side, we'll get a new pending clear when that car leaves.
            if left_clear_matured:
                self._pending_clear_since_left = None
            if right_clear_matured:
                self._pending_clear_since_right = None

        # Update previous state
        self._prev_left = cur_left
        self._prev_right = cur_right

        return calls

    def _cooldown_elapsed(self, call_type: str, current_time: float) -> bool:
        """Check if enough time has passed since the last call of this type.

        Returns True if the call type has never been made (no cooldown to enforce)
        or if the cooldown period has elapsed since the last call.
        """
        # Proximity-related calls use the proximity cooldown
        proximity_types = {"car_left", "car_right", "three_wide"}
        cooldown = (
            self._proximity_cooldown
            if call_type in proximity_types
            else self._clearance_cooldown
        )
        if call_type not in self._last_call_time:
            return True  # Never called before — no cooldown to enforce
        return (current_time - self._last_call_time[call_type]) >= cooldown

    def reset(self):
        """Reset state (e.g. on session start/reconnect)."""
        self._prev_left = False
        self._prev_right = False
        self._last_call_time.clear()
        self._car_alongside = False
        self._pending_clear_since_left = None
        self._pending_clear_since_right = None


class SpotterAudioPlayer:
    """Loads pre-recorded WAV files into memory and plays them on demand.

    Audio is loaded once at startup and played through sounddevice.OutputStream
    instances in daemon threads. This avoids conflicting with TTS playback
    (which uses sd.play/sd.stop) and allows spotter calls to overlap with
    TTS responses.
    """

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._output_device = spotter_config.get("output_device", None)
        self._volume = spotter_config.get("volume", 1.0)
        self._samples: dict[
            str, tuple[np.ndarray, int]
        ] = {}  # key -> (float32_array, sample_rate)

        # Load configured audio files
        audio_paths = spotter_config.get("audio_paths", {})
        for key, path in audio_paths.items():
            self._load_wav(key, path)

        if self._samples:
            logger.info(
                f"Spotter audio loaded: {len(self._samples)} samples "
                f"({', '.join(self._samples.keys())})"
            )
        else:
            logger.warning("No spotter audio samples loaded")

    def _load_wav(self, key: str, path: str):
        """Load a WAV file into memory as a float32 numpy array.

        Args:
            key: Audio sample key (e.g. "car_left", "clear")
            path: Path to the WAV file (relative to project root or absolute)
        """
        if not os.path.exists(path):
            logger.warning(f"Spotter audio file not found: {path} (key: {key})")
            return

        try:
            with wave.open(path, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            # Convert to float32 numpy array
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                logger.warning(
                    f"Unsupported sample width {sample_width} in {path} (key: {key})"
                )
                return

            audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

            # Normalise to [-1.0, 1.0] range
            if dtype == np.int16:
                audio = audio / 32768.0
            elif dtype == np.int32:
                audio = audio / 2147483648.0

            # If stereo, mix down to mono by averaging channels
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            self._samples[key] = (audio, sample_rate)
            duration = len(audio) / sample_rate
            logger.info(f"Loaded spotter audio: {key} = {path} ({duration:.2f}s)")

        except Exception as e:
            logger.warning(f"Failed to load spotter audio {path} (key: {key}): {e}")

    def play(self, key: str):
        """Play a pre-loaded audio sample. Non-blocking.

        Each call spawns a daemon thread that creates its own OutputStream,
        writes the audio, and exits. This allows multiple calls to overlap
        and doesn't interfere with TTS playback.
        """
        if key not in self._samples:
            logger.warning(f"Spotter audio sample not found: {key}")
            return

        audio, sample_rate = self._samples[key]

        # Apply volume at play time (not load time) so config changes take effect
        if self._volume != 1.0:
            audio = audio * self._volume

        # Fire-and-forget in a daemon thread
        thread = threading.Thread(
            target=self._play_audio, args=(audio, sample_rate), daemon=True
        )
        thread.start()

    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Internal: play audio through a dedicated OutputStream.

        Runs in a daemon thread. Creates a new OutputStream per call,
        which is independent of sd.play() / sd.stop() used by TTS.
        """
        try:
            import sounddevice as sd

            # Resolve output device (same logic as TTSClient)
            device = self._resolve_output_device()

            with sd.OutputStream(
                device=device,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                stream.write(audio)

        except ImportError:
            logger.error(
                "sounddevice not installed. Spotter audio disabled. "
                "Install with: uv pip install -e '.[voice]'"
            )
        except Exception as e:
            logger.error(f"Spotter audio playback failed: {e}")

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

    @property
    def available_samples(self) -> list[str]:
        """List of loaded audio sample keys."""
        return list(self._samples.keys())


class Spotter:
    """Top-level spotter coordinator.

    Owns a ProximityDetector and SpotterAudioPlayer. Called once per tick
    from the main loop with the current CarLeftRight value. Decides what
    audio to play based on state transitions.

    Usage:
        spotter = Spotter(config)
        # Each tick:
        spotter.update(car_left_right=1, is_on_track=True)
        # On reconnect:
        spotter.reset()
    """

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._enabled = spotter_config.get("enabled", True)
        self._detector = ProximityDetector(config)
        self._player = SpotterAudioPlayer(config)
        self._config = config

        if self._enabled:
            logger.info("Spotter enabled — car proximity calls active")
            logger.info(f"Available audio samples: {self._player.available_samples}")
        else:
            logger.info("Spotter disabled in config")

    def update(self, car_left_right: int, is_on_track: bool = True):
        """Process a telemetry tick.

        Called once per tick from the main loop. Reads CarLeftRight,
        runs edge detection, and plays any resulting audio calls.

        Args:
            car_left_right: iRacing CarLeftRight value (0=none, 1=left, 2=right, 3=both)
            is_on_track: If False, skip processing (car is in garage/pit stall)
        """
        if not self._enabled:
            return

        # Don't process when not on track (garage, between sessions, etc.)
        if not is_on_track:
            return

        # Guard against invalid values
        if car_left_right < 0 or car_left_right > 3:
            return

        calls = self._detector.update(car_left_right, time.monotonic())
        for call in calls:
            self._player.play(call.audio_key)

    def reset(self):
        """Reset spotter state (on session start/reconnect)."""
        self._detector.reset()
        logger.debug("Spotter state reset")

    @property
    def enabled(self) -> bool:
        """Whether the spotter is enabled."""
        return self._enabled
