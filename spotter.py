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
- Appearance debounce: delays proximity calls by appear_delay_ms to suppress
  telemetry flicker (e.g. iRacing briefly reporting the wrong side).
- Guard: only processes CarLeftRight when the player is on the racing surface
  (PlayerTrackSurface >= 3), suppressing false calls in the pit lane.
- Fuel alert: plays "fuel_two_laps" when estimated laps remaining drops below
  a configurable threshold (default 2.0 laps).
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

    call_type: (
        str  # e.g. "car_left", "car_right", "three_wide", "clear", "fuel_two_laps"
    )
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
    clearance). Cooldown timers prevent repeated calls within a
    configurable window.

    Appearance debounce: when a car first appears on a side, the call is
    delayed by appear_delay_ms. If the reading changes during this window
    (telemetry flicker reporting the wrong side), the pending call is
    updated or cancelled. Only stable readings produce audio calls.

    Clearance debounce: when a car disappears, the "clear" call is delayed
    by clear_delay_ms to suppress telemetry flicker where the car momentarily
    reads as gone.

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
        self._appear_delay = cooldowns.get("appear_delay_ms", 200) / 1000.0

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

        # Pending appearance tracking — debounce to avoid false proximity
        # calls from telemetry flicker (car momentarily reads on wrong side).
        # Set to the monotonic time when the side first appeared;
        # None means the side is not pending an appearance.
        self._pending_appear_since_left: float | None = None
        self._pending_appear_since_right: float | None = None

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

        # === Phase 1: Update pending appearance timers ===
        # When a car first appears on a side, start a pending appear timer
        # instead of firing immediately. If the car disappears or changes
        # side during the appear_delay window, we cancel the pending call.

        if cur_left and not self._prev_left:
            # Car just appeared on left — start pending appear timer
            self._pending_appear_since_left = current_time
        elif not cur_left:
            # Car is not on left — cancel any pending appearance
            self._pending_appear_since_left = None

        if cur_right and not self._prev_right:
            # Car just appeared on right — start pending appear timer
            self._pending_appear_since_right = current_time
        elif not cur_right:
            # Car is not on right — cancel any pending appearance
            self._pending_appear_since_right = None

        # === Phase 2: Check for matured appearance calls ===
        # If a pending appearance has been stable for appear_delay seconds,
        # fire the call. We save the start time before clearing so we can
        # detect three_wide (both sides started pending at the same instant).

        left_appeared = False
        right_appeared = False
        left_appear_start_time: float | None = None
        right_appear_start_time: float | None = None

        if (
            self._pending_appear_since_left is not None
            and cur_left
            and (current_time - self._pending_appear_since_left) >= self._appear_delay
        ):
            # Left appearance has matured — fire it
            left_appeared = True
            left_appear_start_time = self._pending_appear_since_left
            self._pending_appear_since_left = None

        if (
            self._pending_appear_since_right is not None
            and cur_right
            and (current_time - self._pending_appear_since_right) >= self._appear_delay
        ):
            # Right appearance has matured — fire it
            right_appeared = True
            right_appear_start_time = self._pending_appear_since_right
            self._pending_appear_since_right = None

        # === Phase 3: Appearance calls ===
        if left_appeared or right_appeared:
            # Special case: both sides appeared simultaneously from none → three_wide.
            # Detect by checking if both pending timers started at the same instant.
            if (
                left_appeared
                and right_appeared
                and left_appear_start_time == right_appear_start_time
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

        # === Phase 4: Debounced clear detection ===
        # Instead of firing "clear" the instant telemetry reads the car as gone,
        # we require the car to be gone for clear_delay seconds. This prevents
        # false "clear" calls from telemetry flicker.

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
        self._pending_appear_since_left = None
        self._pending_appear_since_right = None


class SpotterAudioPlayer:
    """Loads pre-recorded WAV files into memory and plays them on demand.

    Audio is loaded once at startup and played through sounddevice.OutputStream
    instances in daemon threads. This avoids conflicting with TTS playback
    (which uses sd.play/sd.stop) and allows spotter calls to overlap with
    TTS responses.
    """

    # Default audio paths — can be overridden via config spotter.audio_paths
    DEFAULT_AUDIO_PATHS = {
        "car_left": "audio/carleft.wav",
        "car_right": "audio/carright.wav",
        "three_wide": "audio/carthreewide.wav",
        "clear": "audio/carclear.wav",
        "fuel_five_laps": "audio/fuelfivelaps.wav",
        "fuel_two_laps": "audio/fueltwolaps.wav",
        "fuel_one_lap": "audio/fuelonelap.wav",
        "flag_yellow": "audio/flagyellow.wav",
        "flag_blue": "audio/flagblue.wav",
        "flag_black": "audio/flagblack.wav",
        "flag_white": "audio/flagwhite.wav",
        "flag_red": "audio/flagred.wav",
        "flag_checkered": "audio/flagchequered.wav",
        "flag_slippery": "audio/flagslippery.wav",
        "pit_entry": "audio/pitentry.wav",
        "pit_exit": "audio/pitexit.wav",
    }

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._output_device = spotter_config.get("output_device", None)
        self._volume = spotter_config.get("volume", 1.0)
        self._samples: dict[
            str, tuple[np.ndarray, int]
        ] = {}  # key -> (float32_array, sample_rate)

        # Load configured audio files (overriding defaults where specified)
        audio_paths = {
            **self.DEFAULT_AUDIO_PATHS,
            **spotter_config.get("audio_paths", {}),
        }
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

    Also monitors:
    - Fuel laps remaining (alerts at 5, 2, and 1 lap thresholds)
    - Session flag transitions (yellow, blue, black, white, red, checkered)
    - Track wetness transitions (slippery surface alert)
    - Pit road transitions (pit entry and pit exit)

    Usage:
        spotter = Spotter(config)
        # Each tick:
        spotter.update(car_left_right=1, is_on_track=True, track_surface=3,
                       fuel_laps_remaining=5.2, session_flags=0x01,
                       track_wetness=0.0, on_pit_road=False)
        # On reconnect:
        spotter.reset()
    """

    # iRacing SessionFlags bitmasks (from iRacing SDK Flags class)
    FLAG_CHECKERED = 0x00000001
    FLAG_WHITE = 0x00000010
    FLAG_YELLOW = 0x00000002
    FLAG_RED = 0x00000004
    FLAG_BLACK = 0x00000020
    FLAG_BLUE = 0x00008000

    # Fuel alert thresholds (laps remaining): (threshold, audio_key, reset_threshold)
    # Each alert fires once when laps drop below threshold, resets when laps
    # rise back above reset_threshold (which equals the threshold — simple
    # on/off with no hysteresis).
    FUEL_ALERTS = [
        (5.0, "fuel_five_laps"),
        (2.0, "fuel_two_laps"),
        (1.0, "fuel_one_lap"),
    ]

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._enabled = spotter_config.get("enabled", True)
        self._detector = ProximityDetector(config)
        self._player = SpotterAudioPlayer(config)
        self._config = config

        # Fuel alert state: track which alerts have been fired
        # Key is the threshold value, value is whether it's been fired this stint
        self._fuel_alerts_fired: dict[float, bool] = {
            threshold: False for threshold, _ in self.FUEL_ALERTS
        }

        # Flag state: track previous flags for transition detection
        self._prev_flags: int = 0

        # Track wetness state: previous wetness value for transition detection
        self._prev_track_wetness: float = 0.0
        self._slippery_alert_fired: bool = False

        # Pit road state: track transitions for pit entry/exit alerts
        self._prev_on_pit_road: bool = False

        if self._enabled:
            logger.info("Spotter enabled — car proximity calls active")
            logger.info(f"Available audio samples: {self._player.available_samples}")
        else:
            logger.info("Spotter disabled in config")

    def update(
        self,
        car_left_right: int,
        is_on_track: bool = True,
        track_surface: int = 0,
        fuel_laps_remaining: float = 0.0,
        session_flags: int = 0,
        track_wetness: float = 0.0,
        on_pit_road: bool = False,
    ):
        """Process a telemetry tick.

        Called once per tick from the main loop. Reads CarLeftRight,
        runs edge detection, and plays any resulting audio calls.

        Args:
            car_left_right: iRacing CarLeftRight value (0=none, 1=left, 2=right, 3=both)
            is_on_track: If False, skip processing (car is in garage)
            track_surface: iRacing PlayerTrackSurface value.
                -1=not in world, 0=garage, 1=pit stall, 2=pit road, 3=on racing surface.
                Proximity calls are only processed when >= 3 (on racing surface).
            fuel_laps_remaining: Estimated laps of fuel remaining (0 = unknown).
            session_flags: iRacing SessionFlags bitmask for flag transition detection.
            track_wetness: iRacing TrackWetness value (0-1). Alert fires when
                wetness transitions from 0 to >0 (surface becomes slippery).
            on_pit_road: iRacing OnPitRoad boolean. Alert fires on transitions
                (entering or exiting pit road).
        """
        if not self._enabled:
            return

        # --- Proximity calls (only on racing surface) ---
        if is_on_track and track_surface >= 3 and 0 <= car_left_right <= 3:
            calls = self._detector.update(car_left_right, time.monotonic())
            for call in calls:
                self._player.play(call.audio_key)

        # --- Fuel alerts (always active when on track) ---
        # Fire once when laps remaining drops below each threshold.
        # Reset when laps go back above the threshold (e.g. after pit stop).
        # Don't fire or reset when fuel_laps_remaining is 0 (unknown/unreliable).
        if fuel_laps_remaining > 0:
            for threshold, audio_key in self.FUEL_ALERTS:
                if (
                    fuel_laps_remaining < threshold
                    and not self._fuel_alerts_fired[threshold]
                ):
                    self._player.play(audio_key)
                    self._fuel_alerts_fired[threshold] = True
                    logger.info(
                        f"Fuel alert: {fuel_laps_remaining:.1f} laps remaining ({audio_key})"
                    )
                elif fuel_laps_remaining >= threshold:
                    self._fuel_alerts_fired[threshold] = False

        # --- Flag transition alerts ---
        # Detect rising edges (flag transitions from off to on).
        # Only fire when the player is on track.
        if is_on_track:
            rising = session_flags & ~self._prev_flags  # flags that just turned on
            if rising & self.FLAG_YELLOW:
                self._player.play("flag_yellow")
                logger.info("Flag alert: yellow flag")
            if rising & self.FLAG_BLUE:
                self._player.play("flag_blue")
                logger.info("Flag alert: blue flag")
            if rising & self.FLAG_BLACK:
                self._player.play("flag_black")
                logger.info("Flag alert: black flag")
            if rising & self.FLAG_WHITE:
                self._player.play("flag_white")
                logger.info("Flag alert: white flag")
            if rising & self.FLAG_RED:
                self._player.play("flag_red")
                logger.info("Flag alert: red flag")
            if rising & self.FLAG_CHECKERED:
                self._player.play("flag_checkered")
                logger.info("Flag alert: checkered flag")
            self._prev_flags = session_flags

        # --- Track wetness (slippery surface) alert ---
        # Fire when track wetness transitions from dry (0) to wet (>0).
        # Reset when track dries out (wetness returns to 0).
        # Only fire when on track.
        if is_on_track:
            if (
                track_wetness > 0
                and self._prev_track_wetness == 0
                and not self._slippery_alert_fired
            ):
                self._player.play("flag_slippery")
                self._slippery_alert_fired = True
                logger.info("Track alert: slippery surface (wet)")
            elif track_wetness == 0:
                self._slippery_alert_fired = False
            self._prev_track_wetness = track_wetness

        # --- Pit road transition alerts ---
        # Fire on entering pit road (False→True) and exiting (True→False).
        # Useful for confirming pit entry/exit to the driver.
        if on_pit_road and not self._prev_on_pit_road:
            self._player.play("pit_entry")
            logger.info("Pit road: entered pit road")
        elif not on_pit_road and self._prev_on_pit_road:
            self._player.play("pit_exit")
            logger.info("Pit road: exited pit road")
        self._prev_on_pit_road = on_pit_road

    def reset(self):
        """Reset spotter state (on session start/reconnect)."""
        self._detector.reset()
        for threshold in self._fuel_alerts_fired:
            self._fuel_alerts_fired[threshold] = False
        self._prev_flags = 0
        self._prev_track_wetness = 0.0
        self._slippery_alert_fired = False
        self._prev_on_pit_road = False
        logger.debug("Spotter state reset")

    @property
    def enabled(self) -> bool:
        """Whether the spotter is enabled."""
        return self._enabled
