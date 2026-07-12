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
    CarBehindClosingDetector  ← EMA gap-trend detection for car-behind-closing alert
           │
           ▼
         Spotter           ← coordinator: owns detector + tracker + audio player, called each tick
           │
           ▼
    SpotterAudioPlayer   ← loads WAVs into memory at startup, plays via sounddevice.OutputStream

Key design decisions:
- Uses sounddevice.OutputStream (not sd.play) to avoid killing TTS playback.
- Edge detection, not level detection — fires only on transitions, not every tick.
- Cooldown timers prevent repeated calls within a configurable window.
- Appearance debounce: delays proximity calls by appear_delay_ms to suppress
  telemetry flicker (e.g. iRacing briefly reporting the wrong side).
- Still-there reminder: when a car has been alongside continuously for more
  than still_there_delay_ms (default 5s), plays a "still there" reminder.
  Repeats every still_there_cooldown_ms (default 10s) while the car remains.
- Guard: only processes CarLeftRight when the player is on the racing surface
  (PlayerTrackSurface >= 3), suppressing false calls in the pit lane.
- Fuel alert: plays audio when estimated laps remaining drops below configurable
  thresholds (default: 5, 3, 2, 1 laps).
- Car behind closing: uses lap-time delta comparison (not CarDistBehind derivative)
  for noise-immune detection. Fires after N consecutive faster laps by the car
  behind, when the gap is within configurable bounds, suppressed during yellow.
"""

import logging
import math
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
        str  # e.g. "car_left", "car_right", "three_wide", "clear", "car_still_there"
    )
    priority: int  # 0=safety_critical, 1=team_wide, 2=driver_specific, 3=on_demand
    audio_key: str  # key into the pre-loaded audio samples dict


class ProximityDetector:
    """State machine that detects CarLeftRight transitions.

    iRacing CarLeftRight values (from irsdk.CarLeftRight — ordinal enum, NOT bitmask):
        0 = off (no connection)
        1 = clear (no cars alongside)
        2 = car on the left
        3 = car on the right
        4 = cars on both sides (three wide)
        5 = two cars on the left
        6 = two cars on the right

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
        self._clear_delay = cooldowns.get("clear_delay_ms", 0) / 1000.0
        self._appear_delay = cooldowns.get("appear_delay_ms", 200) / 1000.0
        self._still_there_delay = cooldowns.get("still_there_delay_ms", 5000) / 1000.0
        self._still_there_cooldown = (
            cooldowns.get("still_there_cooldown_ms", 10000) / 1000.0
        )

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

        # Side-correction tracking: when iRacing briefly reports a car on the
        # wrong side before settling on the correct side, we want to correct
        # the wrong-side call by playing "clear" + the correct side.
        # Only applies when the car has been continuously alongside (no gap
        # where CarLeftRight=0) — a gap means genuine car movement, not flicker.
        self._last_side: str | None = None  # "left", "right", or "three_wide"
        self._last_side_time: float = 0.0
        self._side_correction_window: float = 1.5  # seconds
        self._continuous_alongside_since_last_side: bool = False

        # Still-there tracking: monitors how long a car has been continuously
        # alongside. When duration exceeds still_there_delay, a "still there"
        # reminder fires. Repeats every still_there_cooldown while the car
        # remains alongside.
        self._alongside_since: float | None = None  # time when car first appeared
        self._last_still_there_time: float = 0.0  # last time still_there fired

    def update(self, car_left_right: int, current_time: float) -> list[SpotterCall]:
        """Process a CarLeftRight value and return any calls to play.

        Args:
            car_left_right: iRacing CarLeftRight value
                (0=off, 1=clear, 2=left, 3=right, 4=both, 5=two_left, 6=two_right)
            current_time: Monotonic time for cooldown tracking (time.monotonic())

        Returns:
            List of SpotterCall events to play.
        """
        # iRacing CarLeftRight is an ordinal enum (not a bitmask):
        #   0=off, 1=clear, 2=car_left, 3=car_right, 4=both, 5=two_left, 6=two_right
        cur_left = car_left_right in (2, 4, 5)  # car_left, both, two_cars_left
        cur_right = car_left_right in (3, 4, 6)  # car_right, both, two_cars_right

        calls: list[SpotterCall] = []

        # Track whether car has been continuously alongside since the last
        # side call. If CarLeftRight=0 (no car alongside), reset the flag.
        # This distinguishes genuine car movement (car left, gap, car right)
        # from telemetry flicker (car left, briefly wrong side, car left).
        if not cur_left and not cur_right:
            self._continuous_alongside_since_last_side = False

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
                    if self._alongside_since is None:
                        self._alongside_since = current_time
                    self._last_side = "three_wide"
                    self._last_side_time = current_time
                    self._continuous_alongside_since_last_side = True
                    # Reset proximity cooldowns for individual sides too, so
                    # a rapid 0→3→0→1 sequence doesn't re-trigger left immediately
                    self._last_call_time["car_left"] = current_time
                    self._last_call_time["car_right"] = current_time
            else:
                # Individual side appearance — with side-correction for flicker
                if left_appeared:
                    if self._cooldown_elapsed("car_left", current_time):
                        # Side-correction: if the opposite side was just announced
                        # AND the opposite side is no longer present AND there's
                        # been no gap in car-alongside status, this is likely a
                        # telemetry flicker. Play "clear" to correct the wrong
                        # side call, then play the correct side.
                        if (
                            self._last_side == "right"
                            and not cur_right  # right car is no longer there
                            and self._continuous_alongside_since_last_side  # no gap
                            and (current_time - self._last_side_time)
                            < self._side_correction_window
                        ):
                            calls.append(
                                SpotterCall(
                                    call_type="clear", priority=0, audio_key="clear"
                                )
                            )
                        calls.append(
                            SpotterCall(
                                call_type="car_left",
                                priority=0,
                                audio_key="car_left",
                            )
                        )
                        self._last_call_time["car_left"] = current_time
                        self._car_alongside = True
                        if self._alongside_since is None:
                            self._alongside_since = current_time
                        self._last_side = "left"
                        self._last_side_time = current_time
                        self._continuous_alongside_since_last_side = True

                if right_appeared:
                    if self._cooldown_elapsed("car_right", current_time):
                        # Side-correction: mirror of left-side logic above
                        if (
                            self._last_side == "left"
                            and not cur_left  # left car is no longer there
                            and self._continuous_alongside_since_last_side  # no gap
                            and (current_time - self._last_side_time)
                            < self._side_correction_window
                        ):
                            calls.append(
                                SpotterCall(
                                    call_type="clear", priority=0, audio_key="clear"
                                )
                            )
                        calls.append(
                            SpotterCall(
                                call_type="car_right",
                                priority=0,
                                audio_key="car_right",
                            )
                        )
                        self._last_call_time["car_right"] = current_time
                        self._car_alongside = True
                        if self._alongside_since is None:
                            self._alongside_since = current_time
                        self._last_side = "right"
                        self._last_side_time = current_time
                        self._continuous_alongside_since_last_side = True

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
                self._alongside_since = None
                # Only clear pending timers when "clear" actually fires.
                # If we discard them when the guard fails (e.g. cooldown),
                # no new timer can ever start because _prev_left/_prev_right
                # are already False on subsequent ticks, permanently
                # trapping _car_alongside=True and blocking all future clears.
                if left_clear_matured:
                    self._pending_clear_since_left = None
                if right_clear_matured:
                    self._pending_clear_since_right = None
            elif cur_left or cur_right:
                # Car is alongside on another side — suppress "clear" for now.
                # We'll get a new pending clear when that car leaves.
                # Discard the matured timers since they're for a side that's
                # already gone; the remaining side will start its own timer.
                if left_clear_matured:
                    self._pending_clear_since_left = None
                if right_clear_matured:
                    self._pending_clear_since_right = None
            # else: cooldown blocked "clear" — keep pending timers active
            # so they can fire on a subsequent tick when cooldown elapses.

        # === Phase 5: Still-there reminder ===
        # If a car has been continuously alongside for longer than
        # still_there_delay, fire a "still there" reminder. Repeats
        # every still_there_cooldown while the car remains alongside.
        # Guard: also verify cur_left/cur_right so we never fire "still there"
        # when telemetry says no car is alongside — prevents runaway reminders
        # if _car_alongside gets stuck True (e.g. cooldown blocked a "clear").
        if (
            self._alongside_since is not None
            and self._car_alongside
            and (cur_left or cur_right)
            and (current_time - self._alongside_since) >= self._still_there_delay
        ):
            if self._cooldown_elapsed("still_there", current_time):
                calls.append(
                    SpotterCall(
                        call_type="car_still_there",
                        priority=0,
                        audio_key="car_still_there",
                    )
                )
                self._last_call_time["still_there"] = current_time

        # === Phase 6: Safety-net forced clear ===
        # If _car_alongside is True but telemetry shows no car alongside
        # and there's no pending clear timer, the state is stuck — force a
        # clear. This can happen if a matured pending clear was discarded by
        # the cooldown guard before the timer-preservation fix, or due to any
        # other edge case that prevents the normal clear path from firing.
        if (
            self._car_alongside
            and not cur_left
            and not cur_right
            and self._pending_clear_since_left is None
            and self._pending_clear_since_right is None
            and self._cooldown_elapsed("clear", current_time)
        ):
            calls.append(SpotterCall(call_type="clear", priority=0, audio_key="clear"))
            self._last_call_time["clear"] = current_time
            self._car_alongside = False
            self._alongside_since = None

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
        if call_type == "still_there":
            cooldown = self._still_there_cooldown
        elif call_type in proximity_types:
            cooldown = self._proximity_cooldown
        else:
            cooldown = self._clearance_cooldown
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
        self._last_side = None
        self._last_side_time = 0.0
        self._continuous_alongside_since_last_side = False
        self._alongside_since = None
        self._last_still_there_time = 0.0


class CarBehindClosingDetector:
    """Detects when the car behind is genuinely closing on the player.

    Uses iRacing's CarDistBehind (metres) smoothed via EMA to detect
    a shrinking gap. More reliable than lap-time comparison which
    produces false positives when the player has an off-track excursion.

    Two EMAs are maintained:
    - Fast EMA (~1.5s) tracks recent gap changes
    - Slow EMA (~4.0s) tracks the established gap trend

    When the gap is shrinking fast enough (fast_ema < slow_ema by
    more than the threshold), the car is considered closing.
    """

    # Configurable thresholds (can be overridden via config spotter.closing)
    DEFAULT_CLOSING_THRESHOLD_MPS = 0.15  # Min closing rate in m/s to alert
    DEFAULT_EMA_FAST_TAU = 1.5  # Fast EMA time constant (seconds)
    DEFAULT_EMA_SLOW_TAU = 4.0  # Slow EMA time constant (seconds)
    DEFAULT_MAX_GAP = 10.0  # Only alert if gap < this (seconds)
    DEFAULT_MIN_GAP = 1.5  # Don't alert if car is already alongside
    DEFAULT_COOLDOWN = 30.0  # Seconds between repeated alerts

    # Sentinel value for "no car behind" in metres
    NO_CAR_BEHIND = 10000.0

    def __init__(self, config: dict):
        closing_config = config.get("spotter", {}).get("closing", {})
        self._closing_threshold = closing_config.get(
            "closing_threshold_mps", self.DEFAULT_CLOSING_THRESHOLD_MPS
        )
        self._ema_fast_tau = closing_config.get(
            "ema_fast_seconds", self.DEFAULT_EMA_FAST_TAU
        )
        self._ema_slow_tau = closing_config.get(
            "ema_slow_seconds", self.DEFAULT_EMA_SLOW_TAU
        )
        self._max_gap = closing_config.get("max_gap_seconds", self.DEFAULT_MAX_GAP)
        self._min_gap = closing_config.get("min_gap_seconds", self.DEFAULT_MIN_GAP)
        self._cooldown = closing_config.get("cooldown_seconds", self.DEFAULT_COOLDOWN)

        # Warn about deprecated config keys
        for old_key in ("consecutive_faster_laps", "reset_gap_seconds"):
            if old_key in closing_config:
                logger.warning(
                    f"Spotter config 'spotter.closing.{old_key}' is deprecated "
                    f"and ignored — using EMA-based gap detection instead"
                )

        # EMA state (gap in metres)
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._initialized: bool = False
        self._last_update_time: float = 0.0
        self._last_gap_metres: float = 0.0  # For logging

        # Alert state
        self._alert_fired: bool = False
        self._last_alert_time: float = 0.0
        self._car_behind_present: bool = False

    def update(
        self,
        gap_seconds: float,
        gap_metres: float,
        car_on_pit_road: bool,
        player_best_lap_time: float,
        car_behind_lap_time: float,
        is_yellow: bool,
        current_time: float,
    ) -> None:
        """Process a tick of car-behind gap data.

        Args:
            gap_seconds: Absolute gap to car behind in seconds (positive).
                0 means no car behind.
            gap_metres: Gap to car behind in metres from iRacing CarDistBehind.
                Use NO_CAR_BEHIND sentinel if no car behind.
            car_on_pit_road: True if the car behind is on pit road.
            player_best_lap_time: Player's best lap time (0 = no data).
                Used to gate alerts: don't alert if car behind is slower
                than player's best pace.
            car_behind_lap_time: Last lap time of car behind (-1 = no data).
            is_yellow: True if yellow/caution flag is active.
            current_time: Monotonic time for EMA and cooldown tracking.
        """
        # No car behind — reset tracking state
        if gap_seconds <= 0 or gap_metres >= self.NO_CAR_BEHIND:
            if self._car_behind_present:
                logger.debug("Car behind: no car behind, resetting detector")
            self._initialized = False
            self._alert_fired = False
            self._car_behind_present = False
            return

        self._car_behind_present = True

        # Car on pit road — their distance data is unreliable (in/out lap)
        if car_on_pit_road:
            if self._initialized:
                logger.debug("Car behind: on pit road, skipping update")
            return

        # Yellow flag — don't accumulate gap data during caution
        if is_yellow:
            self._initialized = False
            self._alert_fired = False
            return

        # Car alongside or very close — that's CarLeftRight's job
        if gap_seconds < self._min_gap:
            self._initialized = False
            self._alert_fired = False
            return

        # Gap too large — not closing on us meaningfully
        if gap_seconds > self._max_gap:
            self._initialized = False
            self._alert_fired = False
            return

        # Gate: don't alert if car behind is slower than player's best pace
        if player_best_lap_time > 0 and car_behind_lap_time > 0:
            if car_behind_lap_time > player_best_lap_time:
                # Car behind is slower than player's best — not genuinely closing
                self._initialized = False
                self._alert_fired = False
                return

        # Update EMAs with new gap data
        if not self._initialized:
            # First data point — initialise both EMAs to current gap
            self._ema_fast = gap_metres
            self._ema_slow = gap_metres
            self._initialized = True
            self._last_update_time = current_time
            self._last_gap_metres = gap_metres
            return

        # Compute time since last update for EMA alpha
        dt = current_time - self._last_update_time
        if dt <= 0:
            return  # No time elapsed, skip
        self._last_update_time = current_time
        self._last_gap_metres = gap_metres

        # EMA alpha: alpha = 1 - exp(-dt / tau)
        alpha_fast = 1.0 - math.exp(-dt / self._ema_fast_tau)
        alpha_slow = 1.0 - math.exp(-dt / self._ema_slow_tau)

        self._ema_fast = self._ema_fast + alpha_fast * (gap_metres - self._ema_fast)
        self._ema_slow = self._ema_slow + alpha_slow * (gap_metres - self._ema_slow)

        logger.debug(
            f"Car behind: gap={gap_seconds:.1f}s ({gap_metres:.1f}m), "
            f"ema_fast={self._ema_fast:.1f}m, ema_slow={self._ema_slow:.1f}m, "
            f"trend={(self._ema_slow - self._ema_fast) / self._ema_slow_tau:.3f}m/s"
        )

    def should_alert(self, current_time: float | None = None) -> bool:
        """Whether a 'car behind closing' alert should fire this tick.

        Returns True when:
        - Car behind's gap is shrinking (fast EMA < slow EMA by threshold)
        - Gap is within [min_gap, max_gap]
        - Alert hasn't already been fired for this closing approach
        - Cooldown since last alert has elapsed
        - Car behind is faster than player's best pace
        """
        if not self._car_behind_present:
            return False

        if not self._initialized:
            return False

        if self._alert_fired:
            return False

        # Check if gap is shrinking fast enough
        # macd = slow_ema - fast_ema; when positive, gap is shrinking
        macd = self._ema_slow - self._ema_fast
        closing_rate = macd / self._ema_slow_tau  # Approximate m/s closing rate

        if closing_rate < self._closing_threshold:
            return False

        # Cooldown check
        now = current_time if current_time is not None else time.monotonic()
        if self._last_alert_time > 0 and (now - self._last_alert_time) < self._cooldown:
            return False

        # Fire the alert
        self._alert_fired = True
        self._last_alert_time = now
        return True

    def reset(self):
        """Reset all tracking state (e.g. on session start/reconnect)."""
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._initialized = False
        self._last_update_time = 0.0
        self._last_gap_metres = 0.0
        self._alert_fired = False
        self._last_alert_time = 0.0
        self._car_behind_present = False


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
        "fuel_three_laps": "audio/fuelthreelaps.wav",
        "fuel_two_laps": "audio/fueltwolaps.wav",
        "fuel_one_lap": "audio/fuelonelap.wav",
        "flag_yellow": "audio/flagyellow.wav",
        "flag_yellow_waving": "audio/flagyellowwaving.wav",
        "flag_blue": "audio/flagblue.wav",
        "flag_black": "audio/flagblack.wav",
        "flag_black_furled": "audio/flagblackfurled.wav",
        "flag_white": "audio/flagwhite.wav",
        "flag_red": "audio/flagred.wav",
        "flag_checkered": "audio/flagchequered.wav",
        "flag_slippery": "audio/flagslippery.wav",
        "flag_green": "audio/flaggreen.wav",
        "lights_out": "audio/lightsout.wav",
        "flagmeatball": "audio/flagmeatball.wav",
        "penalty_1x": "audio/penaltyonex.wav",
        "penalty_2x": "audio/penaltytwox.wav",
        "penalty_4x": "audio/penaltyfourx.wav",
        "pit_entry": "audio/pitentry.wav",
        "pit_exit": "audio/pitexit.wav",
        "car_behind_closing": "audio/carbehindclosing.wav",
        "car_still_there": "audio/carstillthere.wav",
    }

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._output_device = spotter_config.get("output_device", None)
        self._volume = spotter_config.get("volume", 1.0)
        # Per-key debounce: minimum seconds between plays of the same audio key.
        # Prevents the same call (e.g. "5 laps of fuel") playing twice in
        # rapid succession if the caller logic or telemetry noise triggers it.
        self._debounce_seconds = spotter_config.get("debounce_seconds", 3.0)
        self._last_play_time: dict[str, float] = {}  # key -> monotonic timestamp
        # Global debounce: minimum gap between ANY two audio plays, regardless
        # of key. Prevents overlapping calls (e.g. blue flag + car left)
        # from sounding like two voices at once.
        self._global_debounce_seconds = spotter_config.get(
            "global_debounce_seconds", 0.5
        )
        self._last_play_time_global: float = 0.0
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
            dtype: type = np.int16
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

        A per-key debounce prevents the same audio playing twice in rapid
        succession. A global debounce prevents any two calls from overlapping,
        so the spotter never talks over itself.
        """
        if key not in self._samples:
            logger.warning(f"Spotter audio sample not found: {key}")
            return

        now = time.monotonic()

        # Global debounce: skip if ANY audio played very recently
        if now - self._last_play_time_global < self._global_debounce_seconds:
            logger.debug(
                f"Spotter audio global debounce: {key} "
                f"(last audio {now - self._last_play_time_global:.2f}s ago)"
            )
            return

        # Per-key debounce: skip if same key played very recently
        last = self._last_play_time.get(key, 0.0)
        if now - last < self._debounce_seconds:
            logger.debug(
                f"Spotter audio debounced: {key} (last played {now - last:.2f}s ago)"
            )
            return

        self._last_play_time[key] = now
        self._last_play_time_global = now

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
    - Car behind closing (lap-time delta comparison)

    Usage:
        spotter = Spotter(config)
        # Each tick:
        spotter.update(car_left_right=1, is_on_track=True, track_surface=3,
                       fuel_laps_remaining=5.2, session_flags=0x04,
                       incidents=0, on_pit_road=False,
                       car_behind_gap=-3.5, car_behind_lap_time=91.2,
                       player_last_lap_time=92.5)
        # On reconnect:
        spotter.reset()
    """

    # iRacing SessionFlags bitmasks (from iRacing SDK / irsdk.Flags)
    FLAG_CHECKERED = 0x0001
    FLAG_WHITE = 0x0002
    FLAG_GREEN = 0x0004
    FLAG_YELLOW = 0x0008
    FLAG_RED = 0x0010
    FLAG_BLUE = 0x0020
    FLAG_DEBRIS = 0x0040
    FLAG_CROSSED = 0x0080
    FLAG_YELLOW_WAVING = 0x0100
    FLAG_CAUTION = 0x4000
    FLAG_CAUTION_WAVING = 0x8000
    FLAG_BLACK = 0x010000
    FLAG_DISQUALIFY = 0x020000
    FLAG_SERVICABLE = 0x040000
    FLAG_FURLED = 0x080000
    FLAG_REPAIR = 0x100000

    # Fuel alert thresholds (laps remaining): (threshold, audio_key, hysteresis)
    # Each alert fires once when laps drop below threshold, and only resets
    # when laps rise back above (threshold + hysteresis). This prevents
    # rapid re-triggering when fuel laps remaining oscillates around a
    # threshold (e.g. 5.0 ↔ 4.9 due to iRacing telemetry noise).
    FUEL_ALERTS = [
        (5.0, "fuel_five_laps", 0.5),  # fires at <5.0, resets only at ≥5.5
        (3.0, "fuel_three_laps", 0.5),  # fires at <3.0, resets only at ≥3.5
        (2.0, "fuel_two_laps", 0.3),  # fires at <2.0, resets only at ≥2.3
        (1.0, "fuel_one_lap", 0.3),  # fires at <1.0, resets only at ≥1.3
    ]

    def __init__(self, config: dict):
        spotter_config = config.get("spotter", {})
        self._enabled = spotter_config.get("enabled", True)
        self._detector = ProximityDetector(config)
        self._player = SpotterAudioPlayer(config)
        self._car_behind_tracker = CarBehindClosingDetector(config)
        self._config = config

        # Fuel alert state: track which alerts have been fired
        # Key is the threshold value, value is whether it's been fired this stint
        self._fuel_alerts_fired: dict[float, bool] = {
            threshold: False for threshold, _, _ in self.FUEL_ALERTS
        }
        # Fuel alert cooldown: minimum seconds between repeated fuel alerts
        # of the same type. Safety net against noise even with hysteresis.
        self._fuel_cooldown = spotter_config.get("fuel_cooldown_seconds", 120.0)
        # Track last fire time per threshold for cooldown enforcement.
        # Initial value of None means "never fired" — bypasses cooldown.
        self._fuel_alert_last_time: dict[float, float | None] = {
            threshold: None for threshold, _, _ in self.FUEL_ALERTS
        }

        # Flag state: track previous flags for transition detection
        self._prev_flags: int | None = None  # None = first tick not yet processed
        self._race_started: bool = False  # True when SessionState >= 3
        self._first_lap_completed: bool = (
            False  # True after first lap completed (suppresses blue flag at race start)
        )

        # Session state: track transitions for lights-out detection
        self._prev_session_state: int | None = (
            None  # None = first tick not yet processed
        )
        self._initial_green: bool = False  # True after first green flag of session

        # Incident count state: track previous values for change detection
        self._prev_incidents: int | None = None

        # Pit road state: track transitions for pit entry/exit alerts
        self._prev_on_pit_road: bool | None = (
            None  # None = first tick not yet processed
        )
        self._confirmed_on_pit_road: bool = (
            False  # True only when OnPitRoad + surface agree
        )

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
        session_state: int = 0,
        incidents: int = 0,
        on_pit_road: bool = False,
        car_behind_gap: float = 0.0,
        car_behind_lap_time: float = -1.0,
        car_behind_gap_metres: float = 10000.0,
        player_last_lap_time: float = -1.0,
        player_best_lap_time: float = 0.0,
        car_behind_on_pit_road: bool = False,
        lap_completed: int = 0,
        current_time: float | None = None,
    ):
        """Process a telemetry tick.

        Called once per tick from the main loop. Reads CarLeftRight,
        runs edge detection, and plays any resulting audio calls.

        Args:
            car_left_right: iRacing CarLeftRight value
                (0=off, 1=clear, 2=left, 3=right, 4=both, 5=two_left, 6=two_right)
            is_on_track: If False, skip processing (car is in garage)
            track_surface: iRacing PlayerTrackSurface value.
                -1=not in world, 0=garage, 1=pit stall, 2=pit road, 3=on racing surface.
                Proximity calls are only processed when >= 3 (on racing surface).
            fuel_laps_remaining: Estimated laps of fuel remaining (0 = unknown).
            session_flags: iRacing SessionFlags bitmask for flag transition detection.
            session_state: iRacing SessionState value.
                1=GetInCar, 2=ParadeLaps, 3=Racing, 4=Checkered, 5=CoolDown.
                Used to derive _race_started (state >= 3 means race is on).
            incidents: Player's incident points (1x, 2x, 4x). Alert fires on increases.
            on_pit_road: iRacing OnPitRoad boolean. Alert fires on transitions
                (entering or exiting pit road).
            car_behind_gap: Gap in seconds to the car directly behind (negative
                = behind player). 0 = no car behind.
            car_behind_lap_time: Last lap time of the car directly behind.
                -1.0 = no data available.
            car_behind_gap_metres: Gap to car behind in metres from iRacing
                CarDistBehind. 10000.0 = no car behind (iRacing sentinel).
                Used for EMA-based closing detection.
            player_last_lap_time: Player's last completed lap time.
                -1.0 = no completed lap yet.
            player_best_lap_time: Player's best lap time in this session.
                0.0 = no valid lap yet. Used to gate closing alerts —
                don't alert if car behind is slower than player's best pace.
            car_behind_on_pit_road: True if the car behind is on pit road.
                Their lap time is an in/out lap and not representative.
            lap_completed: Number of laps the player has completed (0 = none yet).
                Used to suppress blue flag at race start — blue flags are not
                meaningful until at least one lap is completed (field hasn't
                spread out yet).
            current_time: Monotonic time for EMA and cooldown tracking.
                Defaults to time.monotonic() if not provided. Pass explicit
                values in tests to simulate time progression.
        """
        if not self._enabled:
            return

        now = current_time if current_time is not None else time.monotonic()

        # Store for use in car-behind-closing logic
        self._player_best_lap_time = player_best_lap_time
        self._car_behind_on_pit_road = car_behind_on_pit_road

        # --- Track first lap completion (blue flag suppression) ---
        if lap_completed >= 1:
            self._first_lap_completed = True

        # --- Session state: lights-out detection + race started ---
        # iRacing SessionState: 1=GetInCar, 2=ParadeLaps, 3=Racing,
        # 4=Checkered, 5=CoolDown.
        # - 2→3 transition = lights out (race goes from parade to racing)
        # - State >= 3 = race is on (sets _race_started for pit/blue alerts)
        # - State drops to 1-2 = new session (clears _race_started)
        # Fallback: if session_state is 0 (not provided), check green flag.
        if self._prev_session_state is None:
            # First tick — prime state
            self._prev_session_state = session_state
            if session_state >= 3:
                self._race_started = True
                logger.info(
                    f"Session state primed: {session_state} (race already started)"
                )
            elif session_state == 0 and session_flags & self.FLAG_GREEN:
                self._race_started = True
                logger.info("Session state primed: green flag fallback")
            else:
                logger.info(f"Session state primed: {session_state}")
        elif session_state != self._prev_session_state:
            if self._prev_session_state in (1, 2) and session_state >= 3:
                # Lights out! GetInCar/ParadeLaps → Racing (or Checkered/CoolDown
                # in some race formats that skip state 3)
                self._race_started = True
                self._player.play("lights_out")
                logger.info(
                    f"Lights out! (SessionState {self._prev_session_state} → {session_state})"
                )
            elif session_state >= 1 and session_state <= 2 and self._race_started:
                # Dropped back to pre-race — new session
                self._race_started = False
                logger.info(f"Session state reset: {session_state} (new session)")
            elif session_state >= 3 and not self._race_started:
                # Joined mid-race at state >= 3 (not via 2→3 transition)
                self._race_started = True
                logger.info(f"Race already started (SessionState={session_state})")
            self._prev_session_state = session_state

        # --- Proximity calls (only on racing surface) ---
        if is_on_track and track_surface >= 3 and 0 <= car_left_right <= 6:
            calls = self._detector.update(car_left_right, now)
            for call in calls:
                self._player.play(call.audio_key)

        # --- Fuel alerts (always active when on track) ---
        # Fire once when laps remaining drops below each threshold.
        # Reset only when laps rise back above (threshold + hysteresis) — this
        # prevents rapid re-triggering when fuel oscillates around a threshold.
        # A per-threshold cooldown also prevents repeated alerts within a stint.
        # Don't fire or reset when fuel_laps_remaining is 0 (unknown/unreliable).
        if fuel_laps_remaining > 0:
            for threshold, audio_key, hysteresis in self.FUEL_ALERTS:
                last_time = self._fuel_alert_last_time[threshold]
                if (
                    fuel_laps_remaining < threshold
                    and not self._fuel_alerts_fired[threshold]
                    and (
                        last_time is None  # never fired
                        or (now - last_time) >= self._fuel_cooldown
                    )
                ):
                    self._player.play(audio_key)
                    self._fuel_alerts_fired[threshold] = True
                    self._fuel_alert_last_time[threshold] = now
                    logger.info(
                        f"Fuel alert: {fuel_laps_remaining:.1f} laps remaining ({audio_key})"
                    )
                elif fuel_laps_remaining >= threshold + hysteresis:
                    self._fuel_alerts_fired[threshold] = False

        # --- Flag transition alerts ---
        # Detect rising edges (flag transitions from off to on).
        # Only fire audio when the player is on track.
        # _race_started is now derived from SessionState (above), not
        # from the green flag edge — so this section only handles audio.
        # First tick: just record the initial flag state, don't alert.
        if self._prev_flags is None:
            # First tick — prime the initial state, no alerts
            self._prev_flags = session_flags
            logger.info(
                f"Flag state primed: 0x{session_flags:08x} (race_started={self._race_started})"
            )
        elif is_on_track:
            rising = session_flags & ~self._prev_flags  # flags that just turned on
            if rising & self.FLAG_YELLOW:
                self._player.play("flag_yellow")
                logger.info("Flag alert: yellow flag")
            if rising & self.FLAG_CAUTION:
                self._player.play("flag_yellow")
                logger.info("Flag alert: caution (yellow)")
            if rising & self.FLAG_CAUTION_WAVING:
                self._player.play("flag_yellow")
                logger.info("Flag alert: caution waving (yellow)")
            if rising & self.FLAG_YELLOW_WAVING:
                self._player.play("flag_yellow_waving")
                logger.info("Flag alert: yellow waving")
            if (
                rising & self.FLAG_BLUE
                and self._race_started
                and self._first_lap_completed
            ):
                self._player.play("flag_blue")
                logger.info("Flag alert: blue flag")
            if rising & self.FLAG_BLACK:
                self._player.play("flag_black")
                logger.info("Flag alert: black flag")
            if rising & self.FLAG_DISQUALIFY:
                self._player.play("flag_black")
                logger.info("Flag alert: disqualify flag")
            if rising & self.FLAG_FURLED:
                self._player.play("flag_black_furled")
                logger.info("Flag alert: furled black flag (warning)")
            if rising & self.FLAG_WHITE:
                self._player.play("flag_white")
                logger.info("Flag alert: white flag")
            if rising & self.FLAG_RED:
                self._player.play("flag_red")
                logger.info("Flag alert: red flag")
            if rising & self.FLAG_CHECKERED:
                self._player.play("flag_checkered")
                logger.info("Flag alert: checkered flag")
            if rising & self.FLAG_DEBRIS:
                self._player.play("flag_slippery")
                logger.info("Flag alert: debris (slippery surface)")
            if rising & self.FLAG_REPAIR:
                self._player.play("flagmeatball")
                logger.info("Flag alert: meatball (repair order)")
            if rising & self.FLAG_GREEN:
                # Absorb current pit road state to avoid false "pit exit"
                # when transitioning from grid to racing surface on green
                self._confirmed_on_pit_road = on_pit_road and track_surface <= 2
                self._prev_on_pit_road = self._confirmed_on_pit_road
                # Skip "green flag" on initial race start — "lights out" already
                # covers that. Only play on subsequent greens (e.g. restarts
                # after yellow). _initial_green tracks whether this is the first
                # green of the session (lights_out already played for it).
                if not self._initial_green:
                    self._initial_green = True
                    logger.info("Flag alert: green flag (initial start, suppressed)")
                else:
                    self._player.play("flag_green")
                    logger.info("Flag alert: green flag (restart)")
            self._prev_flags = session_flags
            if rising:
                logger.debug(
                    f"Flags: prev=0x{self._prev_flags:08x}, "
                    f"curr=0x{session_flags:08x}, rising=0x{rising:08x}"
                )
        else:
            # Not on track — do NOT update _prev_flags so that rising
            # edges are preserved and will fire when the car returns to track.
            # (_race_started is derived from SessionState above, not flags.)
            pass

        # --- Incident count change alerts ---
        # First tick: just record initial state, don't alert.
        if self._prev_incidents is None:
            self._prev_incidents = incidents
            logger.info(f"Incident state primed: incidents={incidents}")
        else:
            # 1x, 2x, 4x penalties: detected when incident points increase.
            if incidents > self._prev_incidents:
                delta = incidents - self._prev_incidents
                if delta >= 4:
                    self._player.play("penalty_4x")
                    logger.info(f"Incident alert: +{delta}x (4x penalty)")
                elif delta == 2:
                    self._player.play("penalty_2x")
                    logger.info("Incident alert: 2x penalty")
                else:
                    self._player.play("penalty_1x")
                    logger.info("Incident alert: 1x penalty")
            self._prev_incidents = incidents

        # --- Pit road transition alerts ---
        # Fire on entering pit road (False→True) and exiting (True→False).
        # Useful for confirming pit entry/exit to the driver.
        # iRacing's OnPitRoad telemetry can flicker True briefly at the
        # start/finish line. To avoid false alerts, we only confirm pit
        # road when OnPitRoad=True AND PlayerTrackSurface indicates pit
        # road or pit stall (values 1-2). If OnPitRoad=True but surface=3
        # (on racing surface), it's a start/finish flicker — ignore it.
        # Also suppress pit alerts before the race goes green — on the grid
        # you're technically "on pit road" but you're not pitting.
        on_pit_road_confirmed = on_pit_road and track_surface <= 2
        if self._prev_on_pit_road is None:
            # First tick — just record the initial state, don't alert
            self._prev_on_pit_road = on_pit_road_confirmed
            self._confirmed_on_pit_road = on_pit_road_confirmed
        else:
            if on_pit_road_confirmed and not self._confirmed_on_pit_road:
                if self._race_started:
                    self._player.play("pit_entry")
                    logger.info("Pit road: entered pit road")
                else:
                    logger.debug("Pit road: on grid (pit entry suppressed)")
            elif not on_pit_road_confirmed and self._confirmed_on_pit_road:
                if self._race_started:
                    self._player.play("pit_exit")
                    logger.info("Pit road: exited pit road")
                else:
                    logger.debug("Pit road: leaving grid (pit exit suppressed)")
            self._confirmed_on_pit_road = on_pit_road_confirmed
            self._prev_on_pit_road = on_pit_road_confirmed

        # --- Car behind closing alert ---
        # Detects when the car directly behind is closing the gap using
        # iRacing's CarDistBehind (metres) smoothed via EMA. This is more
        # reliable than lap-time comparison, which produces false positives
        # when the player has an off-track excursion.
        # Only active when on track, racing, and with valid data.
        if (
            is_on_track
            and self._race_started
            and car_behind_gap < 0  # negative = behind player
            and player_best_lap_time > 0
        ):
            abs_gap = abs(car_behind_gap)
            is_yellow = bool(
                session_flags
                & (
                    self.FLAG_YELLOW
                    | self.FLAG_CAUTION
                    | self.FLAG_CAUTION_WAVING
                    | self.FLAG_YELLOW_WAVING
                )
            )
            # Check if the car behind is on pit road
            car_behind_on_pit = getattr(self, "_car_behind_on_pit_road", False)
            self._car_behind_tracker.update(
                gap_seconds=abs_gap,
                gap_metres=car_behind_gap_metres,
                car_on_pit_road=car_behind_on_pit,
                player_best_lap_time=player_best_lap_time,
                car_behind_lap_time=car_behind_lap_time,
                is_yellow=is_yellow,
                current_time=now,
            )
            if self._car_behind_tracker.should_alert(current_time=now):
                self._player.play("car_behind_closing")
                best = self._player_best_lap_time
                ema_fast = self._car_behind_tracker._ema_fast
                ema_slow = self._car_behind_tracker._ema_slow
                closing_rate = (
                    (ema_slow - ema_fast) / self._car_behind_tracker._ema_slow_tau
                    if self._car_behind_tracker._ema_slow_tau > 0
                    else 0
                )
                logger.info(
                    f"Car behind closing: gap={abs_gap:.1f}s "
                    f"({ema_slow:.1f}m→{ema_fast:.1f}m), "
                    f"trend=-{closing_rate:.2f}m/s, "
                    f"their_lap={car_behind_lap_time:.1f}s, "
                    f"our_best={best:.1f}s"
                )
        elif car_behind_gap >= 0:
            # No car behind — reset tracker
            self._car_behind_tracker.update(
                gap_seconds=0.0,
                gap_metres=CarBehindClosingDetector.NO_CAR_BEHIND,
                car_on_pit_road=False,
                player_best_lap_time=0.0,
                car_behind_lap_time=-1.0,
                is_yellow=False,
                current_time=now,
            )

    def reset(self):
        """Reset spotter state (on session start/reconnect)."""
        self._detector.reset()
        self._car_behind_tracker.reset()
        for threshold in self._fuel_alerts_fired:
            self._fuel_alerts_fired[threshold] = False
        for threshold in self._fuel_alert_last_time:
            self._fuel_alert_last_time[threshold] = None
        self._prev_flags = None
        # _race_started will be re-derived from SessionState on the next
        # tick, so it's correct regardless of whether we reconnect mid-race
        # (state >= 3 → True immediately) or between sessions (state 1-2 → False).
        self._race_started = False
        self._first_lap_completed = False
        self._prev_session_state = None
        self._initial_green = False
        self._prev_incidents = None
        self._prev_on_pit_road = None
        self._confirmed_on_pit_road = False
        self._player._last_play_time.clear()
        self._player._last_play_time_global = 0.0
        logger.debug("Spotter state reset")

    @property
    def enabled(self) -> bool:
        """Whether the spotter is enabled."""
        return self._enabled
