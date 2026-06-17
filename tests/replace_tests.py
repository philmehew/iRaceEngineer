"""Temporary script to replace TestCarBehindTracker with TestCarBehindClosingDetector tests."""

with open("tests/test_spotter.py", "r", encoding="utf-8") as f:
    content = f.read()

old_start = "# CarBehindTracker — Car Behind Closing Alert Tests"
old_end = "\n\nclass TestSpotterCarBehindClosing"

new_tests = r'''# CarBehindClosingDetector — Car Behind Closing Alert Tests
# ---------------------------------------------------------------------------


class TestCarBehindClosingDetector:
    """Test the CarBehindClosingDetector EMA-based gap-trend detection."""

    def setup_method(self):
        self.detector = CarBehindClosingDetector({})

    def _feed_gap_sequence(
        self, gaps_metres, gaps_seconds, times, player_best=98.0, car_lap=95.0
    ):
        """Helper: feed a sequence of gap data and return whether alert fires."""
        alert_fired = False
        for gm, gs, t in zip(gaps_metres, gaps_seconds, times):
            self.detector.update(
                gap_seconds=gs,
                gap_metres=gm,
                car_on_pit_road=False,
                player_best_lap_time=player_best,
                car_behind_lap_time=car_lap,
                is_yellow=False,
                current_time=t,
            )
            if self.detector.should_alert(current_time=t):
                alert_fired = True
        return alert_fired

    def test_no_alert_when_no_car_behind(self):
        """No alert when gap_seconds is 0 (no car behind)."""
        self.detector.update(
            gap_seconds=0.0,
            gap_metres=10000.0,  # sentinel
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=1.0,
        )
        assert not self.detector.should_alert(current_time=1.0)

    def test_no_alert_when_gap_stable(self):
        """No alert when gap is stable (not shrinking)."""
        # Gap stays at ~50m - not closing
        alert = self._feed_gap_sequence(
            gaps_metres=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            gaps_seconds=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        assert not alert

    def test_no_alert_when_gap_growing(self):
        """No alert when gap is growing (car behind is falling back)."""
        # Gap grows from 30m to 65m - car behind is falling back
        alert = self._feed_gap_sequence(
            gaps_metres=[30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            gaps_seconds=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        assert not alert

    def test_alert_when_gap_shrinking_consistently(self):
        """Alert fires when gap is shrinking consistently (car closing)."""
        # Gap shrinks from 50m to 30m over several ticks - car is closing
        alert = self._feed_gap_sequence(
            gaps_metres=[50.0, 48.0, 45.0, 42.0, 39.0, 36.0, 33.0, 30.0],
            gaps_seconds=[5.0, 4.8, 4.5, 4.2, 3.9, 3.6, 3.3, 3.0],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            player_best=98.0,
            car_lap=95.0,  # genuinely faster
        )
        assert alert

    def test_no_alert_when_slower_than_player_best(self):
        """No alert when car behind is slower than player's best lap time."""
        # Gap shrinks, but car behind (1:42) is slower than player's best (1:38)
        alert = self._feed_gap_sequence(
            gaps_metres=[50.0, 48.0, 45.0, 42.0, 39.0, 36.0],
            gaps_seconds=[5.0, 4.8, 4.5, 4.2, 3.9, 3.6],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            player_best=98.0,  # 1:38
            car_lap=102.0,  # 1:42 - slower than player's best
        )
        assert not alert

    def test_no_alert_when_car_on_pit_road(self):
        """Car on pit road should not trigger closing alerts."""
        self.detector.update(
            gap_seconds=5.0,
            gap_metres=50.0,
            car_on_pit_road=True,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=1.0,
        )
        assert not self.detector.should_alert(current_time=1.0)

    def test_no_alert_during_yellow(self):
        """No alert accumulation during yellow flag."""
        self.detector.update(
            gap_seconds=5.0,
            gap_metres=50.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=True,
            current_time=1.0,
        )
        assert not self.detector.should_alert(current_time=1.0)

    def test_no_alert_when_gap_too_small(self):
        """No alert when gap is below min_gap (car alongside - CarLeftRight's job)."""
        # Gap only 0.8s / 8m - car is alongside
        self.detector.update(
            gap_seconds=0.8,
            gap_metres=8.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=1.0,
        )
        self.detector.update(
            gap_seconds=0.7,
            gap_metres=7.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=2.0,
        )
        assert not self.detector.should_alert(current_time=2.0)

    def test_no_alert_when_gap_too_large(self):
        """No alert when gap exceeds max_gap (default 10s)."""
        # Gap 12s / 120m - too far away
        self.detector.update(
            gap_seconds=12.0,
            gap_metres=120.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=1.0,
        )
        self.detector.update(
            gap_seconds=11.0,
            gap_metres=110.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=2.0,
        )
        assert not self.detector.should_alert(current_time=2.0)

    def test_no_repeated_alerts_within_cooldown(self):
        """Alert should not fire again within cooldown period."""
        # Build up closing trend
        alert = self._feed_gap_sequence(
            gaps_metres=[50.0, 48.0, 45.0, 42.0, 39.0, 36.0, 33.0, 30.0],
            gaps_seconds=[5.0, 4.8, 4.5, 4.2, 3.9, 3.6, 3.3, 3.0],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            player_best=98.0,
            car_lap=95.0,
        )
        assert alert

        # Continue closing - should not re-alert within 30s cooldown
        for gm, gs, t in [(28.0, 2.8, 9.0), (26.0, 2.6, 10.0), (24.0, 2.4, 11.0)]:
            self.detector.update(
                gap_seconds=gs,
                gap_metres=gm,
                car_on_pit_road=False,
                player_best_lap_time=98.0,
                car_behind_lap_time=95.0,
                is_yellow=False,
                current_time=t,
            )
            assert not self.detector.should_alert(current_time=t)

    def test_reset_clears_state(self):
        """Reset should clear all EMA state so alerts can re-fire."""
        # Build up closing trend
        alert = self._feed_gap_sequence(
            gaps_metres=[50.0, 48.0, 45.0, 42.0, 39.0, 36.0],
            gaps_seconds=[5.0, 4.8, 4.5, 4.2, 3.9, 3.6],
            times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            player_best=98.0,
            car_lap=95.0,
        )
        assert alert

        self.detector.reset()

        # After reset, must build up EMA again - one tick is not enough
        self.detector.update(
            gap_seconds=5.0,
            gap_metres=50.0,
            car_on_pit_road=False,
            player_best_lap_time=98.0,
            car_behind_lap_time=95.0,
            is_yellow=False,
            current_time=7.0,
        )
        assert not self.detector.should_alert(current_time=7.0)

    def test_custom_thresholds_from_config(self):
        """CarBehindClosingDetector should use custom thresholds from config."""
        config = {
            "spotter": {
                "closing": {
                    "closing_threshold_mps": 0.3,
                    "ema_fast_seconds": 2.0,
                    "ema_slow_seconds": 5.0,
                    "max_gap_seconds": 15.0,
                    "min_gap_seconds": 2.0,
                    "cooldown_seconds": 60.0,
                }
            }
        }
        detector = CarBehindClosingDetector(config)
        assert detector._closing_threshold == 0.3
        assert detector._ema_fast_tau == 2.0
        assert detector._ema_slow_tau == 5.0
        assert detector._max_gap == 15.0
        assert detector._min_gap == 2.0
        assert detector._cooldown == 60.0

    def test_deprecated_config_keys_accepted(self):
        """Deprecated config keys should be accepted without error."""
        config = {
            "spotter": {
                "closing": {
                    "consecutive_faster_laps": 3,  # deprecated
                    "reset_gap_seconds": 15.0,  # deprecated
                    "closing_threshold_mps": 0.2,
                }
            }
        }
        # Should not raise, just log a warning
        detector = CarBehindClosingDetector(config)
        assert detector._closing_threshold == 0.2

'''

# Find the boundaries
start_marker = "# CarBehindTracker — Car Behind Closing Alert Tests"
end_marker = "\n\nclass TestSpotterCarBehindClosing"

start_idx = content.index(start_marker)
end_idx = content.index(end_marker)

# Replace the section
new_content = content[:start_idx] + new_tests + "\n\n" + content[end_idx:]

with open("tests/test_spotter.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Replaced lines {start_idx} to {end_idx} successfully")
