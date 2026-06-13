"""
Unit tests for the spotter module — ProximityDetector state machine.

Tests cover edge detection (appearance and clearance), cooldown enforcement,
clear-delay debounce (flicker suppression), appear-delay debounce (wrong-side
flicker suppression), three-wide special case, reset behaviour, the
track_surface guard, and the fuel alert feature.
"""

from unittest.mock import MagicMock


from spotter import CarBehindTracker, ProximityDetector, Spotter, SpotterCall

# iRacing CarLeftRight enum values (from irsdk.CarLeftRight — ordinal, NOT bitmask)
CLR_OFF = 0
CLR_CLEAR = 1
CLR_CAR_LEFT = 2
CLR_CAR_RIGHT = 3
CLR_BOTH = 4
CLR_TWO_LEFT = 5
CLR_TWO_RIGHT = 6

# Default test config with short delays for fast tests
DEFAULT_CONFIG = {
    "spotter": {
        "cooldowns": {
            "proximity_ms": 3000,
            "clearance_ms": 5000,
            "clear_delay_ms": 100,  # 0.1s — short enough for fast tests
            "appear_delay_ms": 0,  # No appearance delay — immediate calls
            "still_there_delay_ms": 500,  # 0.5s — short for fast tests
            "still_there_cooldown_ms": 1000,  # 1s — short for fast tests
        }
    }
}


def make_detector(config=None):
    """Create a ProximityDetector with default test config."""
    return ProximityDetector(config or DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# ProximityDetector — State Machine Tests
# ---------------------------------------------------------------------------


class TestProximityDetectorTransitions:
    """Test CarLeftRight transition detection."""

    # Map from old bitmask-style values to iRacing enum values:
    #   old 0 (none) → CLR_CLEAR (1), old 1 (left) → CLR_CAR_LEFT (2),
    #   old 2 (right) → CLR_CAR_RIGHT (3), old 3 (both) → CLR_BOTH (4)
    _CLR_MAP = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        """Advance time by dt and process a CarLeftRight value.

        Accepts old-style values (0=none, 1=left, 2=right, 3=both) and
        translates them to iRacing enum values automatically.
        """
        self.t += dt
        mapped = self._CLR_MAP.get(car_lr, car_lr)
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        """Extract call_type strings from a list of SpotterCalls."""
        return [c.call_type for c in calls]

    # --- Appearance calls ---

    def test_no_car_steady_state(self):
        """Staying at 0 should produce no calls."""
        calls = self._tick(0)
        assert calls == []
        calls = self._tick(0)
        assert calls == []

    def test_car_appears_left(self):
        """0→1 should emit car_left."""
        calls = self._tick(0)
        assert calls == []
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]

    def test_car_appears_right(self):
        """0→2 should emit car_right."""
        calls = self._tick(0)
        assert calls == []
        calls = self._tick(2)
        assert self._call_types(calls) == ["car_right"]

    def test_both_sides_from_none(self):
        """0→3 should emit three_wide, not separate car_left + car_right."""
        calls = self._tick(0)
        assert calls == []
        calls = self._tick(3)
        assert self._call_types(calls) == ["three_wide"]

    def test_left_then_right(self):
        """1→3 should emit only car_right (left was already there)."""
        self._tick(1)
        calls = self._tick(3)
        assert self._call_types(calls) == ["car_right"]

    def test_right_then_left(self):
        """2→3 should emit only car_left (right was already there)."""
        self._tick(2)
        calls = self._tick(3)
        assert self._call_types(calls) == ["car_left"]

    def test_left_already_there_stays(self):
        """Staying at 1 should produce no calls after initial transition."""
        self._tick(1)
        calls = self._tick(1)
        assert calls == []

    def test_right_already_there_stays(self):
        """Staying at 2 should produce no calls after initial transition."""
        self._tick(2)
        calls = self._tick(2)
        assert calls == []

    # --- Clearance calls (with debounce) ---

    def test_clear_from_left(self):
        """1→0 should emit clear after the clear delay elapses."""
        self._tick(1)
        # First tick at 0 starts the debounce timer — no clear yet
        calls = self._tick(0)
        assert self._call_types(calls) == []
        # After clear_delay elapses, still at 0, clear fires
        self.t += 0.2  # Past the 100ms clear_delay
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_right(self):
        """2→0 should emit clear after the clear delay elapses."""
        self._tick(2)
        calls = self._tick(0)
        assert self._call_types(calls) == []
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_both(self):
        """3→0 should emit clear after the clear delay elapses."""
        self._tick(3)
        calls = self._tick(0)
        assert self._call_types(calls) == []
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_right_while_left_stays(self):
        """3→1 should NOT emit clear while car still alongside on left."""
        self._tick(3)
        # Right clears, left stays — start debounce
        calls = self._tick(1)
        assert self._call_types(calls) == []
        # After clear_delay, still at 1 (right gone, left still there) —
        # "clear" must NOT fire because a car is still alongside on the left
        self.t += 0.2
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert self._call_types(calls) == []

    def test_clear_left_while_right_stays(self):
        """3→2 should NOT emit clear while car still alongside on right."""
        self._tick(3)
        calls = self._tick(2)
        assert self._call_types(calls) == []
        # After clear_delay, still at 2 (left gone, right still there) —
        # "clear" must NOT fire because a car is still alongside on the right
        self.t += 0.2
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        assert self._call_types(calls) == []

    # --- Combined appearance + clearance ---

    def test_appearance_and_clearance_in_sequence(self):
        """Car appears left, then clears after delay."""
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]
        # Start debounce
        calls = self._tick(0)
        assert self._call_types(calls) == []
        # Clear matures
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_does_not_fire_without_prior_proximity(self):
        """Clear should never fire if no car was ever alongside."""
        # Starting from 0, staying at 0 — no car ever alongside
        self._tick(0)
        self._tick(0)
        self.t += 1.0
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == []

    def test_clear_after_three_wide_then_one_side_clears_then_both_clear(self):
        """3→1→0: clear only fires when ALL cars are gone, not when one side clears."""
        calls = self._tick(3)
        assert self._call_types(calls) == ["three_wide"]
        # Right clears, left stays (3→1) — no "clear" yet
        calls = self._tick(1)
        assert self._call_types(calls) == []
        # After clear_delay, still at 1 — still no "clear" (car on left)
        self.t += 0.2
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert self._call_types(calls) == []
        # Now left also clears (1→0) — pending clear starts for left
        calls = self._tick(0)
        assert self._call_types(calls) == []
        # After clear_delay, clear fires because all cars are gone
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_not_fire_after_proximity_if_car_reappears(self):
        """Clear should not fire if a car reappears before clear matures."""
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]
        # Car disappears (1→0)
        calls = self._tick(0)
        assert self._call_types(calls) == []
        # Car reappears on right before clear matures (0→2)
        self.t += 0.2
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        # Should get car_right, NOT clear
        assert "clear" not in self._call_types(calls)
        assert "car_right" in self._call_types(calls)

    def test_three_wide_then_clear(self):
        """3→0 after three_wide should emit clear after delay."""
        calls = self._tick(3)
        assert self._call_types(calls) == ["three_wide"]
        calls = self._tick(0)
        assert self._call_types(calls) == []
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]


class TestProximityDetectorClearDelay:
    """Test the clear-delay debounce — the core flicker suppression logic."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                }
            }
        }
        self.detector = ProximityDetector(self.config)
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_no_clear_on_brief_flicker(self):
        """If car flickers away for just 1 tick (less than clear_delay),
        no 'clear' call should fire and no re-appearance call either."""
        self._tick(1)  # car left appears
        self._tick(0)  # car flickers away for 1 tick (~100ms, at edge of debounce)
        # Car comes back before clear_delay truly elapsed — flicker suppressed
        self.t += 0.05  # Only 50ms, well within 100ms delay
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        # Should be no calls at all — flicker was suppressed
        assert self._call_types(calls) == []

    def test_clear_fires_after_delay(self):
        """Clear should fire once the car has been gone for clear_delay_ms."""
        self._tick(1)
        self._tick(0)  # Start debounce timer
        # Advance past clear_delay (100ms)
        self.t += 0.15
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_does_not_fire_before_delay(self):
        """Clear should NOT fire if the car has been gone for less than clear_delay."""
        self._tick(1)
        calls = self._tick(0)  # Start debounce timer, but only 100ms tick
        # At this point only 100ms has passed — right at the boundary.
        # We need a bit more time for the delay to truly elapse.
        assert self._call_types(calls) == []

    def test_clear_fires_exactly_at_delay_boundary(self):
        """Clear fires when the delay is exactly met."""
        self._tick(1)
        # Tick at t=0.1 with state=0 starts debounce at t=0.1
        # Need to reach t=0.1 + 0.1 = 0.2 for 100ms delay
        self._tick(0)
        # At t=0.2, 100ms has elapsed since debounce started at t=0.1
        self.t += 0.1
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_flicker_suppresses_both_clear_and_reappearance(self):
        """If car flickers to 0 and back within clear_delay, neither clear
        nor re-appearance should fire — the car never truly left."""
        self._tick(0)  # no car
        calls = self._tick(1)  # car left appears
        assert self._call_types(calls) == ["car_left"]
        # Car flickers away for a single short tick
        calls = self._tick(0)
        assert self._call_types(calls) == []  # debounce, no clear yet
        # Car comes back — this is a flicker, not a real reappearance
        self.t += 0.05
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert self._call_types(calls) == []  # suppressed as flicker

    def test_flicker_then_genuine_clear(self):
        """After a flicker is suppressed, if the car then genuinely leaves,
        the clear should fire after the delay."""
        self._tick(0)
        self._tick(1)  # car left appears
        self._tick(0)  # flicker away (start debounce)
        # Car comes back within delay — flicker suppressed
        self.t += 0.05
        self.detector.update(CLR_CAR_LEFT, self.t)
        # Now car genuinely leaves
        calls = self._tick(0)  # new debounce starts
        assert self._call_types(calls) == []
        # Wait for clear delay
        self.t += 0.15
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_no_double_clear_after_flicker(self):
        """A flicker followed by a genuine clear should produce exactly one
        'clear' call, not two."""
        self._tick(1)  # car appears
        self._tick(0)  # flicker away — debounce starts
        self.t += 0.05
        self.detector.update(CLR_CAR_LEFT, self.t)  # back — flicker suppressed
        self._tick(0)  # genuinely gone again — new debounce
        self.t += 0.15
        calls = self.detector.update(CLR_CLEAR, self.t)
        # Exactly one clear, not two
        assert self._call_types(calls) == ["clear"]
        assert len(calls) == 1

    def test_clear_delay_configurable(self):
        """Clear delay should be configurable via clear_delay_ms."""
        # Use a 200ms delay
        config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 200,
                    "appear_delay_ms": 0,
                }
            }
        }
        detector = ProximityDetector(config)
        t = 0.0

        t += 0.1
        detector.update(CLR_CAR_LEFT, t)  # car appears
        t += 0.1
        detector.update(CLR_CLEAR, t)  # car gone — debounce starts
        # At 150ms, not yet past 200ms delay
        t += 0.15
        calls = detector.update(CLR_CLEAR, t)
        assert [c.call_type for c in calls] == []
        # At 210ms past debounce start, clear should fire
        t += 0.06
        calls = detector.update(CLR_CLEAR, t)
        assert [c.call_type for c in calls] == ["clear"]


class TestProximityDetectorAppearDelay:
    """Test the appear-delay debounce — suppresses wrong-side flicker."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 100,  # 100ms appearance debounce
                }
            }
        }
        self.detector = ProximityDetector(self.config)
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_appearance_delayed(self):
        """Car appearance should be delayed by appear_delay_ms."""
        self._tick(0)  # no car
        calls = self._tick(1)  # car appears on left — but not yet fired (within delay)
        assert self._call_types(calls) == []
        # Next tick: appear_delay has elapsed, call fires
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]

    def test_wrong_side_flicker_suppressed(self):
        """If telemetry briefly flickers to the wrong side, the wrong-side
        call should be suppressed and only the correct side should fire."""
        # Car approaches on left, but telemetry briefly reads 2 (right) for 1 tick
        self._tick(0)  # no car, t=0.1
        calls = self._tick(2)  # flicker: reads right for 1 tick, t=0.2
        assert self._call_types(calls) == []  # not yet fired (within appear_delay)

        # Now telemetry settles to 1 (left) — the car is actually on the left
        calls = self._tick(1)  # t=0.3
        # The "right" pending is cancelled, "left" pending starts
        assert self._call_types(calls) == []  # still within appear_delay for left

        # After appear_delay, left call fires (need 2 more ticks for 100ms to elapse
        # since the left pending started at t=0.3)
        calls = self._tick(1)  # t=0.4 — 0.1s since pending, may or may not fire
        calls = self._tick(1)  # t=0.5 — definitely past 100ms
        assert "car_left" in self._call_types(calls)
        # No "car_right" call was ever made — flicker suppressed!

    def test_appearance_fires_after_delay(self):
        """Appearance should fire once appear_delay has elapsed."""
        self._tick(0)
        self._tick(1)  # car left appears — pending, not yet fired
        # Advance past appear_delay (100ms)
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]

    def test_appear_delay_configurable(self):
        """Appear delay should be configurable via appear_delay_ms."""
        config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 200,  # 200ms debounce
                }
            }
        }
        detector = ProximityDetector(config)
        t = 0.0

        t += 0.1
        detector.update(CLR_CLEAR, t)  # no car
        t += 0.1
        calls = detector.update(CLR_CAR_LEFT, t)  # car left appears — pending
        assert [c.call_type for c in calls] == []

        t += 0.15  # 150ms since appearance — not yet 200ms
        calls = detector.update(CLR_CAR_LEFT, t)
        assert [c.call_type for c in calls] == []

        t += 0.10  # 250ms since appearance — past 200ms
        calls = detector.update(CLR_CAR_LEFT, t)
        assert [c.call_type for c in calls] == ["car_left"]

    def test_three_wide_appearance_delayed(self):
        """Three-wide should also be delayed by appear_delay.
        Both sides start pending at the same time, so when they mature
        together it should emit three_wide (not separate left+right)."""
        self._tick(0)  # no car
        calls = self._tick(3)  # both sides appear — pending
        assert self._call_types(calls) == []

        # After appear_delay, three_wide fires (both matured at same time)
        calls = self._tick(3)
        assert self._call_types(calls) == ["three_wide"]

    def test_appear_cancelled_if_car_disappears(self):
        """If a car appears and then disappears within appear_delay,
        no call should fire and no clear should fire either."""
        self._tick(0)
        self._tick(1)  # car left appears — pending
        # Car disappears within appear_delay
        calls = self._tick(0)
        assert self._call_types(calls) == []

        # No pending appear, no car alongside, no clear should fire
        self.t += 0.5  # well past any delay
        calls = self.detector.update(CLR_CLEAR, self.t)
        # No clear either, since _car_alongside was never set
        assert self._call_types(calls) == []

    def test_reset_clears_pending_appear(self):
        """Reset should cancel pending appearance timers."""
        self._tick(0)
        self._tick(1)  # pending appear for left

        # Reset before delay elapses
        self.detector.reset()

        # After reset, should be able to detect new appearance
        calls = self._tick(0)
        assert calls == []
        calls = self._tick(1)
        # With appear_delay=100ms, this first tick sets pending but doesn't fire
        assert self._call_types(calls) == []
        # Advance well past appear_delay (150ms past the pending start)
        self.t += 0.15
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert self._call_types(calls) == ["car_left"]


class TestProximityDetectorCooldowns:
    """Test cooldown enforcement."""

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_proximity_cooldown_suppresses_rapid_repeat(self):
        """Rapid 0→1→0→1 should suppress the second car_left within cooldown."""
        self._tick(0)
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Car clears (with debounce)
        self._tick(0)
        self.t += 0.2
        self.detector.update(CLR_CLEAR, self.t)  # clear fires

        # Immediately reappear — within 3s cooldown
        calls = self._tick(1)
        assert "car_left" not in self._call_types(calls)

    def test_proximity_cooldown_expires(self):
        """After cooldown expires, same call should fire again."""
        self._tick(0)
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Wait for both clear_delay and proximity cooldown to expire
        self.t += 0.2  # past clear_delay
        self.detector.update(CLR_CLEAR, self.t)  # start clear debounce
        self.t += 3.5  # past proximity cooldown

        # Reappear — should fire again
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_left" in self._call_types(calls)

    def test_clearance_cooldown_suppresses_rapid_repeat(self):
        """Rapid 1→0→1→0 should suppress the second clear within cooldown."""
        self._tick(1)
        # First clear (with debounce)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" in self._call_types(calls)

        # Reappear (may be suppressed by proximity cooldown)
        self.detector.update(CLR_CAR_LEFT, self.t)
        # Second clear attempt (within 5s clearance cooldown)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" not in self._call_types(calls)

    def test_clearance_cooldown_expires(self):
        """After clearance cooldown expires, clear should fire again."""
        self._tick(1)
        # First clear (with debounce)
        self._tick(0)
        self.t += 0.2
        self.detector.update(CLR_CLEAR, self.t)

        # Wait for clearance cooldown to expire (>5s)
        self.t += 5.5

        self.detector.update(CLR_CAR_LEFT, self.t)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" in self._call_types(calls)

    def test_three_wide_resets_individual_cooldowns(self):
        """After three_wide, individual left/right calls should be on cooldown."""
        self._tick(3)  # three_wide fires

        # Clear all sides (with debounce)
        self._tick(0)
        self.t += 0.2
        self.detector.update(CLR_CLEAR, self.t)  # clear fires

        # Immediately reappear on left — should be suppressed by three_wide's cooldown
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_left" not in self._call_types(calls)

    def test_independent_cooldowns_for_left_and_right(self):
        """Left and right have independent cooldowns."""
        self._tick(1)  # car_left fires

        # Immediately appear on right — different side, should fire
        calls = self._tick(3)
        assert "car_right" in self._call_types(calls)


class TestProximityDetectorReset:
    """Test reset behaviour."""

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_reset_clears_state(self):
        """After reset, transitions should fire again even within cooldown period."""
        self._tick(1)  # car_left fires

        # Clear and try to reappear within cooldown
        self._tick(0)

        # Before reset: suppressed
        calls = self._tick(1)
        assert "car_left" not in self._call_types(calls)

        # After reset: should fire again
        self.detector.reset()
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

    def test_reset_clears_previous_state(self):
        """After reset, previous left/right state is cleared."""
        self._tick(3)  # both sides active
        self.detector.reset()

        # After reset, 3 should be treated as a new appearance from 0
        calls = self._tick(3)
        assert "three_wide" in self._call_types(calls)

    def test_reset_clears_pending_clear(self):
        """After reset, pending clear timers should be cancelled."""
        self._tick(1)  # car appears
        self._tick(0)  # start clear debounce

        # Reset before clear matures
        self.detector.reset()

        # Advance time past what would have been the clear delay
        self.t += 0.5
        # Should NOT fire clear — the pending clear was cancelled AND
        # _car_alongside was reset so clear has no basis to fire
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert calls == []

    def test_reset_clears_car_alongside_flag(self):
        """After reset, _car_alongside is cleared so clear won't fire
        even if a pending clear timer had matured before reset."""
        self._tick(1)  # car appears — sets _car_alongside
        self._tick(0)  # start clear debounce
        self.t += 0.5  # advance past clear_delay (timer matures)

        # Reset BEFORE the matured timer is processed
        self.detector.reset()

        # Even though time has passed, clear should not fire because
        # _car_alongside was reset and no car was seen after reset
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert self._call_types(calls) == []

    def test_reset_clears_pending_appear(self):
        """After reset, pending appearance timers should be cancelled."""
        self._tick(1)  # car left appears — pending
        # Reset before delay elapses
        self.detector.reset()
        # Should be a clean slate — no pending appearances
        calls = self._tick(0)
        assert calls == []


class TestProximityDetectorEdgeCases:
    """Test edge cases and invalid inputs."""

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_negative_value_ignored(self):
        """Negative CarLeftRight should be handled gracefully by Spotter.update, not detector."""
        # ProximityDetector doesn't guard — Spotter does
        # Test that detector handles 0 fine
        calls = self._tick(0)
        assert calls == []

    def test_same_value_no_change(self):
        """Repeated same values should produce no calls after initial."""
        calls1 = self._tick(1)
        assert "car_left" in [c.call_type for c in calls1]
        calls2 = self._tick(1)
        assert calls2 == []
        calls3 = self._tick(1)
        assert calls3 == []

    def test_clear_delay_default(self):
        """Default clear_delay_ms should be 0ms (immediate) when not specified in config."""
        config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        detector = ProximityDetector(config)
        assert detector._clear_delay == 0.0  # 0ms default — fire immediately

    def test_appear_delay_default(self):
        """Default appear_delay_ms should be 200ms when not specified in config."""
        config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        detector = ProximityDetector(config)
        assert detector._appear_delay == 0.2  # 200ms default


# ---------------------------------------------------------------------------
# Spotter — Integration Tests (with mocked audio)
# ---------------------------------------------------------------------------


class TestSpotterIntegration:
    """Test Spotter coordinator with mocked audio player."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {
                    "car_left": "audio/carleft.wav",
                    "car_right": "audio/carright.wav",
                    "three_wide": "audio/carthreewide.wav",
                    "clear": "audio/carclear.wav",
                },
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_spotter_skips_when_not_on_track(self):
        """Spotter should not fire when is_on_track=False."""
        spotter = Spotter(self.config)
        # Patch the player to track calls
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(CLR_CAR_LEFT, is_on_track=False)
        assert played_keys == []

    def test_spotter_skips_when_track_surface_below_3(self):
        """Spotter should not fire when PlayerTrackSurface < 3 (not on racing surface).
        Surface values: -1=not in world, 0=garage, 1=pit stall, 2=pit road, 3=on track.
        """
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Surface 0 (garage) — should suppress
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=0)
        assert played_keys == []

        # Surface 1 (pit stall) — should suppress
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=1)
        assert played_keys == []

        # Surface 2 (pit road) — should suppress
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=2)
        assert played_keys == []

        # Surface 3 (on track) — should fire
        spotter.update(CLR_CLEAR, is_on_track=True, track_surface=3)  # reset state
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

    def test_spotter_skips_when_disabled(self):
        """Spotter should not fire when enabled=False in config."""
        self.config["spotter"]["enabled"] = False
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(CLR_CAR_LEFT, is_on_track=True)
        assert played_keys == []

    def test_spotter_fires_on_appearance(self):
        """Spotter should play audio when a car appears on the left."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(CLR_CLEAR, is_on_track=True, track_surface=3)
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

    def test_spotter_reset_clears_state(self):
        """After reset, transitions should fire again."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(CLR_CLEAR, is_on_track=True, track_surface=3)
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

        spotter.reset()

        played_keys.clear()
        spotter.update(CLR_CLEAR, is_on_track=True, track_surface=3)
        spotter.update(CLR_CAR_LEFT, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

    def test_spotter_negative_car_left_right(self):
        """Negative CarLeftRight values should be ignored by Spotter.update."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(-1, is_on_track=True, track_surface=3)
        assert played_keys == []


class TestSpotterFuelAlert:
    """Test fuel alerts at multiple lap thresholds (5, 2, 1)."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_fuel_five_lap_alert(self):
        """Fuel alert should fire at 5-lap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start with plenty of fuel
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=10.0
        )
        assert "fuel_five_laps" not in played_keys

        # Fuel drops below 5
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=4.9
        )
        assert "fuel_five_laps" in played_keys

    def test_fuel_two_lap_alert(self):
        """Fuel alert should fire at 2-lap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.9
        )
        assert "fuel_two_laps" in played_keys

    def test_fuel_one_lap_alert(self):
        """Fuel alert should fire at 1-lap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=0.8
        )
        assert "fuel_one_lap" in played_keys

    def test_fuel_alerts_fire_in_sequence(self):
        """As fuel decreases, each threshold alert fires in order."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start with 10 laps
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=10.0
        )
        # Cross 5-lap threshold
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=4.5
        )
        assert "fuel_five_laps" in played_keys

        played_keys.clear()
        # Cross 2-lap threshold
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys

        played_keys.clear()
        # Cross 1-lap threshold
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=0.5
        )
        assert "fuel_one_lap" in played_keys

    def test_fuel_alert_fires_once(self):
        """Fuel alert should only fire once per fuel window."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys

        # Subsequent ticks should NOT fire again
        played_keys.clear()
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.3
        )
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_resets_above_threshold(self):
        """Fuel alert should reset when laps remaining goes back above threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys

        # Pit stop — fuel goes above threshold
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=5.0
        )

        # Fuel drops below threshold again — should fire again
        played_keys.clear()
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.8
        )
        assert "fuel_two_laps" in played_keys

    def test_fuel_alert_does_not_fire_at_exact_threshold(self):
        """Fuel alert should NOT fire when laps remaining equals the threshold exactly."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=2.0
        )
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_does_not_fire_with_zero(self):
        """Fuel alert should NOT fire when laps remaining is 0 (unknown/unreliable)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=0.0
        )
        assert "fuel_two_laps" not in played_keys
        assert "fuel_five_laps" not in played_keys
        assert "fuel_one_lap" not in played_keys

    def test_fuel_alert_does_not_reset_with_zero(self):
        """Fuel alert should NOT reset when laps remaining is 0 (unknown/unreliable)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys

        # Unknown fuel (0) should NOT reset the alert
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=0.0
        )

        # Should NOT fire again (alert was not reset)
        played_keys.clear()
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.3
        )
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_reset_on_spotter_reset(self):
        """Fuel alert state should be reset when Spotter.reset() is called."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys

        # Reset
        spotter.reset()

        # Should fire again after reset
        played_keys.clear()
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5
        )
        assert "fuel_two_laps" in played_keys


class TestSpotterFlagAlerts:
    """Test flag transition detection (yellow, blue, black, white, red, checkered, debris, repair)."""

    # iRacing SessionFlags bitmasks (from iRacing SDK / irsdk.Flags)
    FLAG_CHECKERED = 0x0001
    FLAG_WHITE = 0x0002
    FLAG_GREEN = 0x0004
    FLAG_YELLOW = 0x0008
    FLAG_RED = 0x0010
    FLAG_BLUE = 0x0020
    FLAG_DEBRIS = 0x0040
    FLAG_CROSSED = 0x0080
    FLAG_CAUTION = 0x4000
    FLAG_CAUTION_WAVING = 0x8000
    FLAG_BLACK = 0x010000
    FLAG_DISQUALIFY = 0x020000
    FLAG_REPAIR = 0x100000

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_yellow_flag_transition(self):
        """Yellow flag should play audio on transition from off to on."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start with green flag (no yellow)
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        assert "flag_yellow" not in played_keys

        # Yellow flag comes out
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

    def test_yellow_flag_no_repeat(self):
        """Yellow flag should not play audio again while already active."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick establishes initial state (no alert)
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Yellow transition
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

        # Still yellow — no repeat
        played_keys.clear()
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" not in played_keys

    def test_yellow_flag_retrigger(self):
        """Yellow flag should play again if it goes off and comes back on."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick establishes initial state
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Yellow on
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

        # Yellow off
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        # Yellow on again
        played_keys.clear()
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

    def test_black_flag_transition(self):
        """Black flag should play audio on transition."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_BLACK,
        )
        assert "flag_black" in played_keys

    def test_white_flag_transition(self):
        """White flag should play audio on transition."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_WHITE,
        )
        assert "flag_white" in played_keys

    def test_blue_flag_transition(self):
        """Blue flag should play audio on transition."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_BLUE,
        )
        assert "flag_blue" in played_keys

    def test_red_flag_transition(self):
        """Red flag should play audio on transition."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_RED,
        )
        assert "flag_red" in played_keys

    def test_checkered_flag_transition(self):
        """Checkered flag should play audio on transition."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_CHECKERED,
        )
        assert "flag_checkered" in played_keys

    def test_flags_not_active_off_track(self):
        """Flag alerts should not fire when not on track."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # is_on_track=False — flags should not fire
        spotter.update(
            0,
            is_on_track=False,
            track_surface=0,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" not in played_keys

    def test_multiple_flags_simultaneous(self):
        """Multiple flags transitioning on at the same time should each play."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW | self.FLAG_WHITE,
        )
        assert "flag_yellow" in played_keys
        assert "flag_white" in played_keys

    def test_flag_alerts_reset_on_spotter_reset(self):
        """Flag state should be reset on Spotter.reset() so flags can re-trigger."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick establishes initial state
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

        spotter.reset()

        # After reset, first tick primes state again (no alert)
        played_keys.clear()
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        assert "flag_yellow" not in played_keys

        # Second tick: yellow should fire
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys


class TestSpotterSlipperyAlert:
    """Test debris flag / slippery surface alert."""

    # iRacing SessionFlags bitmasks (from iRacing SDK / irsdk.Flags)
    FLAG_GREEN = 0x0004
    FLAG_DEBRIS = 0x0040

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_slippery_alert_on_debris_flag(self):
        """Slippery alert should fire when debris flag transitions on."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start with green flag only
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        assert "flag_slippery" not in played_keys

        # Debris flag comes out
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys

    def test_slippery_alert_no_repeat(self):
        """Slippery alert should not fire again while debris flag stays on."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick establishes initial state
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Debris transition
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys

        # Still debris — no repeat
        played_keys.clear()
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" not in played_keys

    def test_slippery_alert_resets_when_debris_cleared(self):
        """Slippery alert should re-fire when debris flag goes off and comes back."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick establishes initial state
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Debris on
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys

        # Debris off
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Debris on again
        played_keys.clear()
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys

    def test_slippery_alert_not_when_off_track(self):
        """Slippery alert should not fire when not on track."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(
            0,
            is_on_track=False,
            track_surface=0,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" not in played_keys

    def test_slippery_alert_resets_on_spotter_reset(self):
        """Slippery alert should re-fire after Spotter.reset()."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys

        spotter.reset()

        # After reset, first tick primes state again
        played_keys.clear()
        spotter.update(
            0, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        assert "flag_slippery" not in played_keys

        # Second tick: debris should fire again
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_DEBRIS,
        )
        assert "flag_slippery" in played_keys


class TestSpotterPitRoadAlerts:
    """Test pit road entry/exit transition alerts."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_pit_entry_transition(self):
        """Pit entry should play audio when OnPitRoad and track_surface confirm pit road."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: first tick sets initial state (no alert on first tick)
        # Pass green flag so race_started=True
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )
        assert "pit_entry" not in played_keys

        # Enter pit road — surface confirms (pit road = 2)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" in played_keys

    def test_pit_exit_transition(self):
        """Pit exit should play audio when OnPitRoad and track_surface confirm exit."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: start on pit road (first tick, no alert)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" not in played_keys  # first tick, no transition

        # Exit pit road — surface confirms (on track = 3)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )
        assert "pit_exit" in played_keys

    def test_pit_road_no_repeat(self):
        """Pit entry/exit should not fire on consecutive same-state ticks."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: not on pit road
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )

        # Enter pit road
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" in played_keys

        # Still on pit road — no repeat
        played_keys.clear()
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" not in played_keys
        assert "pit_exit" not in played_keys

    def test_pit_alerts_reset_on_spotter_reset(self):
        """Pit state should be reset on Spotter.reset() so transitions re-trigger."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: not on pit road (green flag = race started)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )

        # Enter pit road
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" in played_keys

        spotter.reset()

        # After reset, first tick primes state again (no alert)
        played_keys.clear()
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )
        assert "pit_entry" not in played_keys

        # Enter pit road again after reset
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" in played_keys

    def test_pit_road_start_finish_flicker(self):
        """OnPitRoad flicker at start/finish should not trigger pit alerts."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: on racing surface, not on pit road (green flag = race started)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )
        assert "pit_entry" not in played_keys

        # iRacing flicker: OnPitRoad=True but still on racing surface (3)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=True,
        )
        assert "pit_entry" not in played_keys  # surface disagrees — flicker
        assert "pit_exit" not in played_keys

        # Flicker ends: back to normal
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0x0004,
            on_pit_road=False,
        )
        assert "pit_entry" not in played_keys
        assert "pit_exit" not in played_keys

    def test_pit_road_suppressed_before_green(self):
        """Pit entry/exit should be suppressed before green flag (on grid)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: on grid, no green flag yet
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=2,
            session_flags=0,
            on_pit_road=True,
        )
        assert "pit_entry" not in played_keys  # on grid, not racing

        # Cross start line — leave "pit road" area
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=0,
            on_pit_road=False,
        )
        assert "pit_exit" not in played_keys  # still no green flag


# ---------------------------------------------------------------------------
# SpotterAudioPlayer — WAV Loading Tests
# ---------------------------------------------------------------------------


class TestSpotterAudioPlayer:
    """Test audio file loading (doesn't require audio hardware)."""

    def test_missing_file_logged_as_warning(self):
        """Missing audio files should be skipped with a warning, not crash."""
        config = {
            "spotter": {
                "audio_paths": {
                    "car_left": "nonexistent.wav",
                },
                "output_device": None,
                "volume": 1.0,
            }
        }
        player = MagicMock()
        # Just verify the SpotterAudioPlayer can be created without crashing
        # even when audio files don't exist
        from spotter import SpotterAudioPlayer

        player = SpotterAudioPlayer(config)
        # No samples should be loaded (file doesn't exist)
        assert "car_left" not in player.available_samples

    def test_available_samples_list(self):
        """available_samples should list loaded sample keys (not missing ones)."""
        from spotter import SpotterAudioPlayer

        config = {
            "spotter": {
                "audio_paths": {
                    "car_left": "nonexistent.wav",
                },
                "output_device": None,
                "volume": 1.0,
            }
        }
        player = SpotterAudioPlayer(config)
        # car_left was overridden to a nonexistent file, so it shouldn't be loaded
        # Other default paths (that have real files) should still be present
        assert "car_left" not in player.available_samples

    def test_play_missing_key_does_not_crash(self):
        """Playing a missing key should log warning but not crash."""
        from spotter import SpotterAudioPlayer

        config = {
            "spotter": {
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }
        player = SpotterAudioPlayer(config)
        # Should not raise
        player.play("nonexistent_key")

    def test_load_real_wav_files(self):
        """Test loading actual WAV files from the audio directory."""
        from spotter import SpotterAudioPlayer

        config = {
            "spotter": {
                "audio_paths": {
                    "car_left": "audio/carleft.wav",
                    "car_right": "audio/carright.wav",
                },
                "output_device": None,
                "volume": 1.0,
            }
        }
        player = SpotterAudioPlayer(config)
        assert "car_left" in player.available_samples
        assert "car_right" in player.available_samples

    def test_default_audio_paths_include_all_alerts(self):
        """Default audio paths should include all alert types."""
        from spotter import SpotterAudioPlayer

        expected_keys = [
            "car_left",
            "car_right",
            "three_wide",
            "clear",
            "fuel_five_laps",
            "fuel_two_laps",
            "fuel_one_lap",
            "flag_yellow",
            "flag_blue",
            "flag_black",
            "flag_white",
            "flag_red",
            "flag_checkered",
            "flag_green",
            "lights_out",
            "flag_slippery",
            "flagmeatball",
            "penalty_1x",
            "penalty_2x",
            "penalty_4x",
            "pit_entry",
            "pit_exit",
            "car_behind_closing",
            "car_still_there",
        ]
        for key in expected_keys:
            assert key in SpotterAudioPlayer.DEFAULT_AUDIO_PATHS, (
                f"Missing default path: {key}"
            )

    def test_default_audio_paths_include_car_behind_closing(self):
        """Default audio paths should include car_behind_closing alert."""
        from spotter import SpotterAudioPlayer

        assert "car_behind_closing" in SpotterAudioPlayer.DEFAULT_AUDIO_PATHS


# ---------------------------------------------------------------------------
# CarBehindTracker — Car Behind Closing Alert Tests
# ---------------------------------------------------------------------------


class TestCarBehindTracker:
    """Test the CarBehindTracker lap-time delta detection."""

    def setup_method(self):
        self.tracker = CarBehindTracker({})

    def test_no_alert_when_no_car_behind(self):
        """No alert when gap_seconds is 0 (no car behind)."""
        self.tracker.update(
            gap_seconds=0.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        assert not self.tracker.should_alert(current_time=1.0)

    def test_no_alert_when_car_behind_is_slower(self):
        """No alert when the car behind is running slower laps."""
        # Lap 1: car behind is slower
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=False, is_yellow=False, current_time=1.0
        )
        assert not self.tracker.should_alert(current_time=1.0)
        # Lap 2: still slower
        self.tracker.update(
            gap_seconds=5.5, car_behind_faster=False, is_yellow=False, current_time=2.0
        )
        assert not self.tracker.should_alert(current_time=2.0)

    def test_alert_after_consecutive_faster_laps(self):
        """Alert fires after car behind is faster for 2 consecutive laps within gap."""
        # Lap 1: car behind is faster
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        assert not self.tracker.should_alert(current_time=1.0)  # only 1 lap faster

        # Lap 2: car behind is faster again
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)  # 2 laps faster — fire!

    def test_no_alert_when_gap_too_large(self):
        """No alert when gap exceeds max_gap threshold (default 10s)."""
        # Car behind is faster but too far away
        self.tracker.update(
            gap_seconds=15.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=14.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert not self.tracker.should_alert(current_time=2.0)

    def test_no_alert_when_gap_too_small(self):
        """No alert when gap is below min_gap (car already alongside — CarLeftRight's job)."""
        # Car behind is faster and very close (alongside)
        self.tracker.update(
            gap_seconds=1.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=0.8, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert not self.tracker.should_alert(current_time=2.0)

    def test_alert_resets_when_gap_grows(self):
        """Alert resets when gap grows above reset threshold (12s default)."""
        # Lap 1 & 2: car behind is faster — alert fires
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)

        # Gap grows beyond reset threshold — tracker resets
        self.tracker.update(
            gap_seconds=13.0, car_behind_faster=False, is_yellow=False, current_time=3.0
        )

        # Car behind closes again — should be able to fire again
        # Advance time past the 30s cooldown
        self.tracker.update(
            gap_seconds=8.0, car_behind_faster=True, is_yellow=False, current_time=40.0
        )
        self.tracker.update(
            gap_seconds=7.0, car_behind_faster=True, is_yellow=False, current_time=41.0
        )
        assert self.tracker.should_alert(
            current_time=41.0
        )  # Can re-alert after gap grew

    def test_alert_resets_when_car_behind_slows(self):
        """Alert resets when car behind stops being faster (slower lap)."""
        # Lap 1 & 2: faster — alert fires
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)

        # Car behind slows down — resets alert_fired and consecutive count
        self.tracker.update(
            gap_seconds=4.5, car_behind_faster=False, is_yellow=False, current_time=3.0
        )

        # Must rebuild consecutive count (time past cooldown)
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=40.0
        )
        assert not self.tracker.should_alert(
            current_time=40.0
        )  # Only 1 consecutive faster lap

    def test_no_alert_during_yellow(self):
        """No alert accumulation during yellow flag (field closure bunched up)."""
        # Lap 1: faster, but yellow flag
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=True, current_time=1.0
        )
        # Lap 2: faster, still yellow
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=True, current_time=2.0
        )
        assert not self.tracker.should_alert(
            current_time=2.0
        )  # Yellow suppresses counting

        # After yellow clears, must start counting fresh
        self.tracker.update(
            gap_seconds=3.0, car_behind_faster=True, is_yellow=False, current_time=3.0
        )
        assert not self.tracker.should_alert(
            current_time=3.0
        )  # Only 1 lap after yellow

    def test_no_repeated_alerts(self):
        """Alert should not fire again until reset conditions are met."""
        # Build up to alert
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)

        # Continuing ticks should not re-alert (within 30s cooldown)
        self.tracker.update(
            gap_seconds=3.5, car_behind_faster=True, is_yellow=False, current_time=3.0
        )
        assert not self.tracker.should_alert(current_time=3.0)
        self.tracker.update(
            gap_seconds=3.0, car_behind_faster=True, is_yellow=False, current_time=4.0
        )
        assert not self.tracker.should_alert(current_time=4.0)

    def test_reset_clears_state(self):
        """Reset should clear all tracking state so alerts can re-fire."""
        # Build up to alert
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)

        self.tracker.reset()

        # After reset, must build up consecutive count again
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=3.0
        )
        assert not self.tracker.should_alert(current_time=3.0)  # Only 1 lap

    def test_no_car_behind_resets_tracker(self):
        """Passing gap_seconds=0 resets tracker state completely."""
        # Build up to alert
        self.tracker.update(
            gap_seconds=5.0, car_behind_faster=True, is_yellow=False, current_time=1.0
        )
        self.tracker.update(
            gap_seconds=4.0, car_behind_faster=True, is_yellow=False, current_time=2.0
        )
        assert self.tracker.should_alert(current_time=2.0)

        # Car behind pits/disappears
        self.tracker.update(
            gap_seconds=0.0, car_behind_faster=False, is_yellow=False, current_time=3.0
        )

        # Must rebuild consecutive count
        self.tracker.update(
            gap_seconds=6.0, car_behind_faster=True, is_yellow=False, current_time=4.0
        )
        assert not self.tracker.should_alert(current_time=4.0)

    def test_custom_thresholds_from_config(self):
        """CarBehindTracker should use custom thresholds from config."""
        config = {
            "spotter": {
                "closing": {
                    "consecutive_faster_laps": 3,
                    "max_gap_seconds": 15.0,
                    "min_gap_seconds": 2.0,
                    "reset_gap_seconds": 18.0,
                    "cooldown_seconds": 60.0,
                }
            }
        }
        tracker = CarBehindTracker(config)
        assert tracker._consecutive_threshold == 3
        assert tracker._max_gap == 15.0
        assert tracker._min_gap == 2.0
        assert tracker._reset_gap == 18.0
        assert tracker._cooldown == 60.0


class TestSpotterCarBehindClosing:
    """Integration test for car-behind-closing alert through the Spotter class."""

    # iRacing SessionFlags bitmasks
    FLAG_GREEN = 0x0004
    FLAG_YELLOW = 0x0008

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                },
                "audio_paths": {},
                "output_device": None,
                "volume": 1.0,
            }
        }

    def test_no_alert_without_car_behind_data(self):
        """No alert when car_behind_gap=0 (default — no data)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start green flag — race is on
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Tick with no car behind data — should not crash or alert
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=0.0,
            car_behind_lap_time=-1.0,
            player_last_lap_time=-1.0,
        )
        assert "car_behind_closing" not in played_keys

    def test_car_behind_closing_alert_fires(self):
        """Alert fires when car behind is consistently faster within gap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: green flag, on track
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Tick 1: car behind is faster (gap = 5s, their lap = 91s, our lap = 92s)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-5.0,  # negative = behind
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" not in played_keys  # 1st faster lap, not enough

        # Tick 2: car behind still faster (gap shrinking)
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-4.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" in played_keys  # 2 consecutive faster laps

    def test_no_alert_when_not_on_track(self):
        """No alert when player is not on track."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Not on track — no alert even though car behind is closing
        spotter.update(
            CLR_CLEAR,
            is_on_track=False,  # NOT on track
            track_surface=0,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-5.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=False,
            track_surface=0,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-4.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" not in played_keys

    def test_no_alert_during_yellow_flag(self):
        """No closing alert during yellow flag (field bunched up)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime: green flag
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Yellow flag — car behind is faster but yellow suppresses
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
            car_behind_gap=-5.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
            car_behind_gap=-4.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" not in played_keys

    def test_no_alert_when_car_alongside(self):
        """No closing alert when gap < 1.5s (car is alongside — CarLeftRight's job)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )

        # Gap only 0.8s — car alongside, not closing
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-0.8,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-0.8,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" not in played_keys

    def test_reset_clears_car_behind_tracker(self):
        """Spotter.reset() should clear car-behind tracker state."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Prime and trigger alert
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-5.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-4.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" in played_keys

        # Reset
        spotter.reset()

        # Should need to rebuild consecutive count after reset
        played_keys.clear()
        spotter.update(
            CLR_CLEAR, is_on_track=True, track_surface=3, session_flags=self.FLAG_GREEN
        )
        spotter.update(
            CLR_CLEAR,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN,
            car_behind_gap=-5.0,
            car_behind_lap_time=91.0,
            player_last_lap_time=92.0,
        )
        assert "car_behind_closing" not in played_keys  # Only 1 lap, need 2


class TestProximityDetectorStillThere:
    """Test the 'still there' reminder — fires when a car has been alongside
    continuously for more than still_there_delay_ms, repeats every
    still_there_cooldown_ms, and resets when the car clears."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                    "still_there_delay_ms": 500,  # 0.5s for fast tests
                    "still_there_cooldown_ms": 1000,  # 1s for fast tests
                }
            }
        }
        self.detector = ProximityDetector(self.config)
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        mapped = {0: CLR_CLEAR, 1: CLR_CAR_LEFT, 2: CLR_CAR_RIGHT, 3: CLR_BOTH}.get(
            car_lr, car_lr
        )
        return self.detector.update(mapped, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_still_there_fires_after_delay(self):
        """'Still there' should fire when a car has been alongside for
        longer than still_there_delay."""
        # Car appears left
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Stay alongside for 0.4s — not yet past the 0.5s delay
        for _ in range(4):
            calls = self._tick(1)
            assert "car_still_there" not in self._call_types(calls)

        # Stay alongside for 0.1s more — total 0.5s, past the delay
        self.t += 0.1
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_still_there" in self._call_types(calls)

    def test_still_there_not_before_delay(self):
        """'Still there' should NOT fire before the delay has elapsed."""
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Stay alongside for 0.4s — not yet past the 0.5s delay
        for _ in range(4):
            calls = self._tick(1)
            assert "car_still_there" not in self._call_types(calls)

    def test_still_there_resets_on_clear(self):
        """'Still there' timer should reset when the car clears."""
        # Car appears left
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Stay alongside for 0.4s (not yet past delay)
        for _ in range(4):
            self._tick(1)

        # Car clears
        self._tick(0)
        self.t += 0.2  # past clear_delay
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" in self._call_types(calls)

        # Advance past proximity cooldown (3s) so car_left can fire again
        self.t += 3.5

        # Car reappears — timer should restart
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_left" in self._call_types(calls)

        # Stay alongside for 0.4s — not yet past the delay (timer restarted)
        for _ in range(4):
            calls = self._tick(1)
            assert "car_still_there" not in self._call_types(calls)

    def test_still_there_repeats_with_cooldown(self):
        """'Still there' should repeat every still_there_cooldown while
        the car remains alongside."""
        # Car appears left
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Wait for still_there_delay (0.5s) — first still_there
        self.t += 0.5
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_still_there" in self._call_types(calls)

        # Wait for cooldown (1s) — second still_there
        self.t += 1.0
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_still_there" in self._call_types(calls)

    def test_still_there_does_not_repeat_within_cooldown(self):
        """'Still there' should NOT repeat within the cooldown period."""
        # Car appears left
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Wait for still_there_delay (0.5s) — first still_there
        self.t += 0.5
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_still_there" in self._call_types(calls)

        # 0.5s later — still within cooldown (1s), should not fire
        self.t += 0.5
        calls = self.detector.update(CLR_CAR_LEFT, self.t)
        assert "car_still_there" not in self._call_types(calls)

    def test_still_there_for_car_right(self):
        """'Still there' should also work for car on the right."""
        calls = self._tick(2)
        assert "car_right" in self._call_types(calls)

        # Wait for still_there_delay
        self.t += 0.5
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        assert "car_still_there" in self._call_types(calls)

    def test_still_there_for_three_wide(self):
        """'Still there' should work for three-wide (both sides)."""
        calls = self._tick(3)
        assert "three_wide" in self._call_types(calls)

        # Wait for still_there_delay
        self.t += 0.5
        calls = self.detector.update(CLR_BOTH, self.t)
        assert "car_still_there" in self._call_types(calls)

    def test_still_there_not_fired_when_no_car_alongside(self):
        """'Still there' should never fire when no car is alongside."""
        # No car alongside
        calls = self._tick(0)
        assert calls == []

        # Advance well past the delay
        self.t += 10.0
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "car_still_there" not in self._call_types(calls)

    def test_still_there_timer_persists_through_side_switch(self):
        """'Still there' timer should persist when car switches sides
        (side correction), since the car has been continuously alongside."""
        # Car appears left
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Stay alongside for 0.3s
        for _ in range(3):
            self._tick(1)

        # Car switches to right (side correction scenario)
        # The car is still alongside, just on a different side
        # Since we have appear_delay=0, the car_right fires immediately
        # but _alongside_since should persist from the original car_left call
        original_alongside_since = self.detector._alongside_since

        calls = self._tick(2)
        # car_right should fire (with side correction clear+right)
        # _alongside_since should NOT be reset
        assert self.detector._alongside_since == original_alongside_since

    def test_still_there_resets_on_spotter_reset(self):
        """Spotter.reset() should clear still-there timer state."""
        self._tick(1)  # car left appears

        # Advance partway through the delay
        for _ in range(3):
            self._tick(1)

        # Reset
        self.detector.reset()

        # Car appears again — timer should restart from scratch
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Only 0.1s since appearance — not yet past delay
        calls = self._tick(1)
        assert "car_still_there" not in self._call_types(calls)

    def test_still_there_delay_configurable(self):
        """still_there_delay should be configurable via still_there_delay_ms."""
        config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                    "still_there_delay_ms": 1000,  # 1s delay
                    "still_there_cooldown_ms": 5000,
                }
            }
        }
        detector = ProximityDetector(config)
        t = 0.0

        # Car appears left
        t += 0.1
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_left" in [c.call_type for c in calls]

        # 0.5s — not yet past 1s delay
        t += 0.5
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_still_there" not in [c.call_type for c in calls]

        # 0.6s more — total 1.2s, past the 1s delay
        t += 0.6
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_still_there" in [c.call_type for c in calls]

    def test_still_there_cooldown_configurable(self):
        """still_there_cooldown should be configurable via still_there_cooldown_ms."""
        config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,
                    "clear_delay_ms": 100,
                    "appear_delay_ms": 0,
                    "still_there_delay_ms": 200,
                    "still_there_cooldown_ms": 5000,  # 5s cooldown
                }
            }
        }
        detector = ProximityDetector(config)
        t = 0.0

        # Car appears left
        t += 0.1
        detector.update(CLR_CAR_LEFT, t)

        # Wait for delay — first still_there
        t += 0.2
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_still_there" in [c.call_type for c in calls]

        # 2s later — within 5s cooldown, should not fire
        t += 2.0
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_still_there" not in [c.call_type for c in calls]

        # 3.1s more — total 5.1s past last still_there, past cooldown
        t += 3.1
        calls = detector.update(CLR_CAR_LEFT, t)
        assert "car_still_there" in [c.call_type for c in calls]

    def test_still_there_default_values(self):
        """Default still_there_delay should be 5s, cooldown 10s."""
        config = {"spotter": {"cooldowns": {}}}
        detector = ProximityDetector(config)
        assert detector._still_there_delay == 5.0  # 5000ms default
        assert detector._still_there_cooldown == 10.0  # 10000ms default

    def test_still_there_does_not_fire_after_car_clears(self):
        """'Still there' must NOT fire when telemetry shows the car has cleared,
        even if _car_alongside is stuck True. Regression test: the still-there
        check now verifies cur_left/cur_right, not just internal state."""
        # Car appears right
        calls = self._tick(2)
        assert "car_right" in self._call_types(calls)

        # Stay alongside past still_there_delay (0.5s)
        self.t += 0.6
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        assert "car_still_there" in self._call_types(calls)

        # Car clears — need two ticks: first starts pending clear timer,
        # second (after clear_delay elapses) fires the clear
        self.t += 0.05
        calls = self.detector.update(CLR_CLEAR, self.t)  # starts pending clear
        self.t += 0.15  # past 100ms clear_delay
        calls = self.detector.update(CLR_CLEAR, self.t)  # clear matures
        assert "clear" in self._call_types(calls)

        # Advance time — still-there must NOT fire (no car alongside)
        self.t += 1.0
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "car_still_there" not in self._call_types(calls)


class TestProximityDetectorClearCooldownBug:
    """Regression tests for the bug where the clearance cooldown could
    permanently block 'clear' calls, trapping _car_alongside=True forever.

    Bug scenario:
      1. Car alongside → proximity call, _car_alongside=True
      2. Car clears → 'clear' fires, clearance cooldown starts (5s)
      3. Second car appears within cooldown → proximity call, _car_alongside=True
      4. Second car clears within cooldown → pending clear matures but cooldown
         blocks it. Old code discarded the pending timer, so no future clear
         could ever fire (_prev_right already False, no new timer starts).
      5. Result: _car_alongside stuck True, 'still there' fires forever.
    """

    def setup_method(self):
        self.config = {
            "spotter": {
                "cooldowns": {
                    "proximity_ms": 3000,
                    "clearance_ms": 5000,  # 5s clearance cooldown
                    "clear_delay_ms": 0,  # No debounce on clear
                    "appear_delay_ms": 0,
                    "still_there_delay_ms": 500,
                    "still_there_cooldown_ms": 1000,
                }
            }
        }
        self.detector = ProximityDetector(self.config)
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        return self.detector.update(car_lr, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_clear_fires_after_cooldown_blocks_first_attempt(self):
        """When clearance cooldown blocks a clear, the pending timer must be
        preserved so clear can fire on a subsequent tick when cooldown elapses."""
        # Car 1: appears right, clears — establishes clearance cooldown
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)

        self.t += 0.2
        calls = self._tick(CLR_CLEAR)
        assert "clear" in self._call_types(calls)

        # Advance past proximity cooldown (3s) so car_right can fire again
        self.t += 3.0

        # Car 2: appears right WITHIN the 5s clearance cooldown
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)

        # Car 2 clears — still within clearance cooldown (only ~3.4s since
        # first clear). Clear is blocked by cooldown, but pending timer
        # MUST persist so clear can fire later.
        calls = self._tick(CLR_CLEAR)
        # Clear may be blocked by cooldown — that's expected

        # Advance past the 5s clearance cooldown from the first clear
        self.t += 2.0
        calls = self.detector.update(CLR_CLEAR, self.t)
        # Clear MUST fire now — cooldown has elapsed, no car alongside,
        # and pending timer was preserved (not discarded)
        assert "clear" in self._call_types(calls)

    def test_still_there_stops_when_car_clears_despite_cooldown(self):
        """'Still there' must stop when the car clears, even if the clearance
        cooldown temporarily blocks the 'clear' audio call. The telemetry
        check (cur_left/cur_right) prevents runaway reminders."""
        # Car appears right
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)

        # Stay alongside past still_there_delay
        self.t += 0.6
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        assert "car_still_there" in self._call_types(calls)

        # Car clears (even if 'clear' is blocked by cooldown, still-there
        # must not fire because cur_left/cur_right are both False)
        calls = self._tick(CLR_CLEAR)
        # Regardless of whether clear fires, still-there must stop
        self.t += 1.0
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "car_still_there" not in self._call_types(calls)

    def test_safety_net_forced_clear(self):
        """If _car_alongside gets stuck True with no car alongside and no
        pending clear timer, the safety-net forced clear must fire to
        unstick the state."""
        # Car appears right
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)
        assert self.detector._car_alongside is True

        # Car clears — with clear_delay=0, clear fires on same tick
        calls = self._tick(CLR_CLEAR)
        assert "clear" in self._call_types(calls)
        assert self.detector._car_alongside is False

    def test_clear_retries_after_cooldown_when_car_switches_sides(self):
        """When car switches sides (left→right) and the side-clear is suppressed
        because a car is still alongside, the pending timer for the cleared
        side is correctly discarded and a new timer starts when the remaining
        car leaves."""
        # Car on left
        calls = self._tick(CLR_CAR_LEFT)
        assert "car_left" in self._call_types(calls)

        # Car switches to both sides (three wide briefly)
        self.t += 0.1
        calls = self.detector.update(CLR_BOTH, self.t)
        # May or may not get three_wide depending on timing/state

        # Car clears from left but remains on right
        self.t += 0.2
        calls = self.detector.update(CLR_CAR_RIGHT, self.t)
        # Left clear pending timer should be discarded because car is
        # still alongside on the right

        # Car clears from right
        self.t += 5.0  # past clearance cooldown
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" in self._call_types(calls)

    def test_no_stuck_state_after_rapid_car_appearance_and_clear(self):
        """Rapid car appearance and clear within clearance cooldown must not
        leave _car_alongside stuck True."""
        # First car appears and clears
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)

        self.t += 0.1
        calls = self.detector.update(CLR_CLEAR, self.t)
        assert "clear" in self._call_types(calls)

        # Advance past proximity cooldown (3s) so second car_right can fire
        self.t += 3.0

        # Second car appears and clears rapidly (within clearance cooldown)
        calls = self._tick(CLR_CAR_RIGHT)
        assert "car_right" in self._call_types(calls)

        # Second car clears — blocked by clearance cooldown, but timer preserved
        calls = self._tick(CLR_CLEAR)

        # Advance past clearance cooldown — clear must fire
        self.t += 5.0
        calls = self.detector.update(CLR_CLEAR, self.t)

        # _car_alongside must be False — not stuck
        assert self.detector._car_alongside is False

        # No still-there should fire
        assert "car_still_there" not in self._call_types(calls)
