"""
Unit tests for the spotter module — ProximityDetector state machine.

Tests cover edge detection (appearance and clearance), cooldown enforcement,
clear-delay debounce (flicker suppression), appear-delay debounce (wrong-side
flicker suppression), three-wide special case, reset behaviour, the
track_surface guard, and the fuel alert feature.
"""

from unittest.mock import MagicMock

import pytest

from spotter import ProximityDetector, Spotter, SpotterCall

# Default test config with short delays for fast tests
DEFAULT_CONFIG = {
    "spotter": {
        "cooldowns": {
            "proximity_ms": 3000,
            "clearance_ms": 5000,
            "clear_delay_ms": 100,  # 0.1s — short enough for fast tests
            "appear_delay_ms": 0,  # No appearance delay — immediate calls
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

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        """Advance time by dt and process a CarLeftRight value."""
        self.t += dt
        return self.detector.update(car_lr, self.t)

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
        calls = self.detector.update(0, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_right(self):
        """2→0 should emit clear after the clear delay elapses."""
        self._tick(2)
        calls = self._tick(0)
        assert self._call_types(calls) == []
        self.t += 0.2
        calls = self.detector.update(0, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_both(self):
        """3→0 should emit clear after the clear delay elapses."""
        self._tick(3)
        calls = self._tick(0)
        assert self._call_types(calls) == []
        self.t += 0.2
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(1, self.t)
        assert self._call_types(calls) == []

    def test_clear_left_while_right_stays(self):
        """3→2 should NOT emit clear while car still alongside on right."""
        self._tick(3)
        calls = self._tick(2)
        assert self._call_types(calls) == []
        # After clear_delay, still at 2 (left gone, right still there) —
        # "clear" must NOT fire because a car is still alongside on the right
        self.t += 0.2
        calls = self.detector.update(2, self.t)
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
        calls = self.detector.update(0, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_clear_does_not_fire_without_prior_proximity(self):
        """Clear should never fire if no car was ever alongside."""
        # Starting from 0, staying at 0 — no car ever alongside
        self._tick(0)
        self._tick(0)
        self.t += 1.0
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(1, self.t)
        assert self._call_types(calls) == []
        # Now left also clears (1→0) — pending clear starts for left
        calls = self._tick(0)
        assert self._call_types(calls) == []
        # After clear_delay, clear fires because all cars are gone
        self.t += 0.2
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(2, self.t)
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
        calls = self.detector.update(0, self.t)
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
        return self.detector.update(car_lr, self.t)

    def _call_types(self, calls: list[SpotterCall]) -> list[str]:
        return [c.call_type for c in calls]

    def test_no_clear_on_brief_flicker(self):
        """If car flickers away for just 1 tick (less than clear_delay),
        no 'clear' call should fire and no re-appearance call either."""
        self._tick(1)  # car left appears
        self._tick(0)  # car flickers away for 1 tick (~100ms, at edge of debounce)
        # Car comes back before clear_delay truly elapsed — flicker suppressed
        self.t += 0.05  # Only 50ms, well within 100ms delay
        calls = self.detector.update(1, self.t)
        # Should be no calls at all — flicker was suppressed
        assert self._call_types(calls) == []

    def test_clear_fires_after_delay(self):
        """Clear should fire once the car has been gone for clear_delay_ms."""
        self._tick(1)
        self._tick(0)  # Start debounce timer
        # Advance past clear_delay (100ms)
        self.t += 0.15
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(1, self.t)
        assert self._call_types(calls) == []  # suppressed as flicker

    def test_flicker_then_genuine_clear(self):
        """After a flicker is suppressed, if the car then genuinely leaves,
        the clear should fire after the delay."""
        self._tick(0)
        self._tick(1)  # car left appears
        self._tick(0)  # flicker away (start debounce)
        # Car comes back within delay — flicker suppressed
        self.t += 0.05
        self.detector.update(1, self.t)
        # Now car genuinely leaves
        calls = self._tick(0)  # new debounce starts
        assert self._call_types(calls) == []
        # Wait for clear delay
        self.t += 0.15
        calls = self.detector.update(0, self.t)
        assert self._call_types(calls) == ["clear"]

    def test_no_double_clear_after_flicker(self):
        """A flicker followed by a genuine clear should produce exactly one
        'clear' call, not two."""
        self._tick(1)  # car appears
        self._tick(0)  # flicker away — debounce starts
        self.t += 0.05
        self.detector.update(1, self.t)  # back — flicker suppressed
        self._tick(0)  # genuinely gone again — new debounce
        self.t += 0.15
        calls = self.detector.update(0, self.t)
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
        detector.update(1, t)  # car appears
        t += 0.1
        detector.update(0, t)  # car gone — debounce starts
        # At 150ms, not yet past 200ms delay
        t += 0.15
        calls = detector.update(0, t)
        assert [c.call_type for c in calls] == []
        # At 210ms past debounce start, clear should fire
        t += 0.06
        calls = detector.update(0, t)
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
        return self.detector.update(car_lr, self.t)

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
        detector.update(0, t)  # no car
        t += 0.1
        calls = detector.update(1, t)  # car left appears — pending
        assert [c.call_type for c in calls] == []

        t += 0.15  # 150ms since appearance — not yet 200ms
        calls = detector.update(1, t)
        assert [c.call_type for c in calls] == []

        t += 0.10  # 250ms since appearance — past 200ms
        calls = detector.update(1, t)
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
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(1, self.t)
        assert self._call_types(calls) == ["car_left"]


class TestProximityDetectorCooldowns:
    """Test cooldown enforcement."""

    def setup_method(self):
        self.detector = make_detector()
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        return self.detector.update(car_lr, self.t)

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
        self.detector.update(0, self.t)  # clear fires

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
        self.detector.update(0, self.t)  # start clear debounce
        self.t += 3.5  # past proximity cooldown

        # Reappear — should fire again
        calls = self.detector.update(1, self.t)
        assert "car_left" in self._call_types(calls)

    def test_clearance_cooldown_suppresses_rapid_repeat(self):
        """Rapid 1→0→1→0 should suppress the second clear within cooldown."""
        self._tick(1)
        # First clear (with debounce)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(0, self.t)
        assert "clear" in self._call_types(calls)

        # Reappear (may be suppressed by proximity cooldown)
        self.detector.update(1, self.t)
        # Second clear attempt (within 5s clearance cooldown)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(0, self.t)
        assert "clear" not in self._call_types(calls)

    def test_clearance_cooldown_expires(self):
        """After clearance cooldown expires, clear should fire again."""
        self._tick(1)
        # First clear (with debounce)
        self._tick(0)
        self.t += 0.2
        self.detector.update(0, self.t)

        # Wait for clearance cooldown to expire (>5s)
        self.t += 5.5

        self.detector.update(1, self.t)
        self._tick(0)
        self.t += 0.2
        calls = self.detector.update(0, self.t)
        assert "clear" in self._call_types(calls)

    def test_three_wide_resets_individual_cooldowns(self):
        """After three_wide, individual left/right calls should be on cooldown."""
        self._tick(3)  # three_wide fires

        # Clear all sides (with debounce)
        self._tick(0)
        self.t += 0.2
        self.detector.update(0, self.t)  # clear fires

        # Immediately reappear on left — should be suppressed by three_wide's cooldown
        calls = self.detector.update(1, self.t)
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
        return self.detector.update(car_lr, self.t)

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
        calls = self.detector.update(0, self.t)
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
        calls = self.detector.update(0, self.t)
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
        return self.detector.update(car_lr, self.t)

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
        """Default clear_delay_ms should be 500ms when not specified in config."""
        config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        detector = ProximityDetector(config)
        assert detector._clear_delay == 0.5  # 500ms default

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

        spotter.update(1, is_on_track=False)
        assert played_keys == []

    def test_spotter_skips_when_track_surface_below_3(self):
        """Spotter should not fire when PlayerTrackSurface < 3 (not on racing surface).
        Surface values: -1=not in world, 0=garage, 1=pit stall, 2=pit road, 3=on track.
        """
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Surface 0 (garage) — should suppress
        spotter.update(1, is_on_track=True, track_surface=0)
        assert played_keys == []

        # Surface 1 (pit stall) — should suppress
        spotter.update(1, is_on_track=True, track_surface=1)
        assert played_keys == []

        # Surface 2 (pit road) — should suppress
        spotter.update(1, is_on_track=True, track_surface=2)
        assert played_keys == []

        # Surface 3 (on track) — should fire
        spotter.update(0, is_on_track=True, track_surface=3)  # reset state
        spotter.update(1, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

    def test_spotter_skips_when_disabled(self):
        """Spotter should not fire when enabled=False in config."""
        self.config["spotter"]["enabled"] = False
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(1, is_on_track=True)
        assert played_keys == []

    def test_spotter_fires_on_appearance(self):
        """Spotter should play audio when a car appears on the left."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3)
        spotter.update(1, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

    def test_spotter_reset_clears_state(self):
        """After reset, transitions should fire again."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3)
        spotter.update(1, is_on_track=True, track_surface=3)
        assert "car_left" in played_keys

        spotter.reset()

        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3)
        spotter.update(1, is_on_track=True, track_surface=3)
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
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=10.0)
        assert "fuel_five_laps" not in played_keys

        # Fuel drops below 5
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=4.9)
        assert "fuel_five_laps" in played_keys

    def test_fuel_two_lap_alert(self):
        """Fuel alert should fire at 2-lap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.9)
        assert "fuel_two_laps" in played_keys

    def test_fuel_one_lap_alert(self):
        """Fuel alert should fire at 1-lap threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=0.8)
        assert "fuel_one_lap" in played_keys

    def test_fuel_alerts_fire_in_sequence(self):
        """As fuel decreases, each threshold alert fires in order."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start with 10 laps
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=10.0)
        # Cross 5-lap threshold
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=4.5)
        assert "fuel_five_laps" in played_keys

        played_keys.clear()
        # Cross 2-lap threshold
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys

        played_keys.clear()
        # Cross 1-lap threshold
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=0.5)
        assert "fuel_one_lap" in played_keys

    def test_fuel_alert_fires_once(self):
        """Fuel alert should only fire once per fuel window."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys

        # Subsequent ticks should NOT fire again
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.3)
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_resets_above_threshold(self):
        """Fuel alert should reset when laps remaining goes back above threshold."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys

        # Pit stop — fuel goes above threshold
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=5.0)

        # Fuel drops below threshold again — should fire again
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.8)
        assert "fuel_two_laps" in played_keys

    def test_fuel_alert_does_not_fire_at_exact_threshold(self):
        """Fuel alert should NOT fire when laps remaining equals the threshold exactly."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=2.0)
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_does_not_fire_with_zero(self):
        """Fuel alert should NOT fire when laps remaining is 0 (unknown/unreliable)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=0.0)
        assert "fuel_two_laps" not in played_keys
        assert "fuel_five_laps" not in played_keys
        assert "fuel_one_lap" not in played_keys

    def test_fuel_alert_does_not_reset_with_zero(self):
        """Fuel alert should NOT reset when laps remaining is 0 (unknown/unreliable)."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys

        # Unknown fuel (0) should NOT reset the alert
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=0.0)

        # Should NOT fire again (alert was not reset)
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.3)
        assert "fuel_two_laps" not in played_keys

    def test_fuel_alert_reset_on_spotter_reset(self):
        """Fuel alert state should be reset when Spotter.reset() is called."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Fire alert
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys

        # Reset
        spotter.reset()

        # Should fire again after reset
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, fuel_laps_remaining=1.5)
        assert "fuel_two_laps" in played_keys


class TestSpotterFlagAlerts:
    """Test flag transition detection (yellow, blue, black, white, red, checkered)."""

    # iRacing SessionFlags bitmasks (same as Spotter.FLAG_* constants)
    FLAG_CHECKERED = 0x00000001
    FLAG_GREEN = 0x00000001
    FLAG_YELLOW = 0x00000002
    FLAG_RED = 0x00000004
    FLAG_WHITE = 0x00000010
    FLAG_BLACK = 0x00000020
    FLAG_BLUE = 0x00008000

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

        # First transition
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

        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys

        spotter.reset()

        # After reset, yellow should fire again (prev_flags was reset)
        played_keys.clear()
        spotter.update(
            0,
            is_on_track=True,
            track_surface=3,
            session_flags=self.FLAG_GREEN | self.FLAG_YELLOW,
        )
        assert "flag_yellow" in played_keys


class TestSpotterSlipperyAlert:
    """Test track wetness / slippery surface alert."""

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

    def test_slippery_alert_on_wet_transition(self):
        """Slippery alert should fire when track wetness goes from 0 to >0."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start dry
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.0)
        assert "flag_slippery" not in played_keys

        # Track gets wet
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.3)
        assert "flag_slippery" in played_keys

    def test_slippery_alert_no_repeat(self):
        """Slippery alert should not fire again while track stays wet."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # First wet transition
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.3)
        assert "flag_slippery" in played_keys

        # Still wet — no repeat
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.5)
        assert "flag_slippery" not in played_keys

    def test_slippery_alert_resets_when_dry(self):
        """Slippery alert should reset when track dries out and re-fire on next wet."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Wet
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.3)
        assert "flag_slippery" in played_keys

        # Dry
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.0)

        # Wet again
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.2)
        assert "flag_slippery" in played_keys

    def test_slippery_alert_not_when_off_track(self):
        """Slippery alert should not fire when not on track."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=False, track_surface=0, track_wetness=0.5)
        assert "flag_slippery" not in played_keys

    def test_slippery_alert_resets_on_spotter_reset(self):
        """Slippery alert state should be reset on Spotter.reset()."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.3)
        assert "flag_slippery" in played_keys

        spotter.reset()

        # After reset, wet transition should fire again (prev_wetness was reset to 0)
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, track_wetness=0.3)
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
        """Pit entry should play audio when OnPitRoad transitions to True."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Not on pit road
        spotter.update(0, is_on_track=True, track_surface=3, on_pit_road=False)
        assert "pit_entry" not in played_keys

        # Enter pit road
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" in played_keys

    def test_pit_exit_transition(self):
        """Pit exit should play audio when OnPitRoad transitions to False."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Start on pit road
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" in played_keys

        # Exit pit road
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=3, on_pit_road=False)
        assert "pit_exit" in played_keys

    def test_pit_road_no_repeat(self):
        """Pit entry/exit should not fire on consecutive same-state ticks."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Enter pit road
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" in played_keys

        # Still on pit road — no repeat
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" not in played_keys
        assert "pit_exit" not in played_keys

    def test_pit_alerts_reset_on_spotter_reset(self):
        """Pit state should be reset on Spotter.reset() so transitions re-trigger."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        # Enter pit road
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" in played_keys

        spotter.reset()

        # After reset, entering pit road should fire again
        played_keys.clear()
        spotter.update(0, is_on_track=True, track_surface=2, on_pit_road=True)
        assert "pit_entry" in played_keys


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
            "flag_black",
            "flag_white",
            "flag_slippery",
        ]
        for key in expected_keys:
            assert key in SpotterAudioPlayer.DEFAULT_AUDIO_PATHS, (
                f"Missing default path: {key}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
