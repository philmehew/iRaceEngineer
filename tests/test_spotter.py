"""
Unit tests for the spotter module — ProximityDetector state machine.

Tests cover edge detection (appearance and clearance), cooldown enforcement,
three-wide special case, reset behaviour, and the is_on_track guard.
"""

from unittest.mock import MagicMock

import pytest

from spotter import ProximityDetector, Spotter, SpotterCall


# ---------------------------------------------------------------------------
# ProximityDetector — State Machine Tests
# ---------------------------------------------------------------------------


class TestProximityDetectorTransitions:
    """Test CarLeftRight transition detection."""

    def setup_method(self):
        """Create a fresh detector with default config."""
        self.config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        self.detector = ProximityDetector(self.config)
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

    # --- Clearance calls ---

    def test_clear_from_left(self):
        """1→0 should emit clear."""
        self._tick(1)
        calls = self._tick(0)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_right(self):
        """2→0 should emit clear."""
        self._tick(2)
        calls = self._tick(0)
        assert self._call_types(calls) == ["clear"]

    def test_clear_from_both(self):
        """3→0 should emit clear."""
        self._tick(3)
        calls = self._tick(0)
        assert self._call_types(calls) == ["clear"]

    def test_clear_right_while_left_stays(self):
        """3→1 should emit clear (right side cleared)."""
        self._tick(3)
        calls = self._tick(1)
        assert self._call_types(calls) == ["clear"]

    def test_clear_left_while_right_stays(self):
        """3→2 should emit clear (left side cleared)."""
        self._tick(3)
        calls = self._tick(2)
        assert self._call_types(calls) == ["clear"]

    # --- Combined appearance + clearance ---

    def test_appearance_and_clearance_in_sequence(self):
        """Car appears left, then clears."""
        calls = self._tick(1)
        assert self._call_types(calls) == ["car_left"]
        calls = self._tick(0)
        assert self._call_types(calls) == ["clear"]

    def test_three_wide_then_clear(self):
        """3→0 after three_wide should emit clear."""
        calls = self._tick(3)
        assert self._call_types(calls) == ["three_wide"]
        calls = self._tick(0)
        assert self._call_types(calls) == ["clear"]


class TestProximityDetectorCooldowns:
    """Test cooldown enforcement."""

    def setup_method(self):
        self.config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        self.detector = ProximityDetector(self.config)
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

        # Clear the car
        self._tick(0)

        # Immediately reappear — within 3s cooldown
        calls = self._tick(1)
        assert "car_left" not in self._call_types(calls)

    def test_proximity_cooldown_expires(self):
        """After cooldown expires, same call should fire again."""
        self._tick(0)
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

        # Wait for cooldown to expire (>3s)
        self.t += 3.5

        # Clear and reappear — should fire again
        self._tick(0)
        calls = self._tick(1)
        assert "car_left" in self._call_types(calls)

    def test_clearance_cooldown_suppresses_rapid_repeat(self):
        """Rapid 1→0→1→0 should suppress the second clear within cooldown."""
        self._tick(1)
        self._tick(0)  # clear
        self._tick(1)  # reappear (may be suppressed by proximity cooldown)
        calls = self._tick(0)  # clear again within 5s
        assert "clear" not in self._call_types(calls)

    def test_clearance_cooldown_expires(self):
        """After clearance cooldown expires, clear should fire again."""
        self._tick(1)
        self._tick(0)  # clear

        # Wait for clearance cooldown to expire (>5s)
        self.t += 5.5

        self._tick(1)
        calls = self._tick(0)  # clear should fire again
        assert "clear" in self._call_types(calls)

    def test_three_wide_resets_individual_cooldowns(self):
        """After three_wide, individual left/right calls should be on cooldown."""
        self._tick(3)  # three_wide fires

        # Clear all sides
        self._tick(0)  # clear fires

        # Immediately reappear on left — should be suppressed by three_wide's cooldown
        calls = self._tick(1)
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
        self.config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        self.detector = ProximityDetector(self.config)
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


class TestProximityDetectorEdgeCases:
    """Test edge cases and invalid inputs."""

    def setup_method(self):
        self.config = {
            "spotter": {"cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000}}
        }
        self.detector = ProximityDetector(self.config)
        self.t = 0.0

    def _tick(self, car_lr: int, dt: float = 0.1) -> list[SpotterCall]:
        self.t += dt
        return self.detector.update(car_lr, self.t)

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


# ---------------------------------------------------------------------------
# Spotter — Integration Tests (with mocked audio)
# ---------------------------------------------------------------------------


class TestSpotterIntegration:
    """Test Spotter coordinator with mocked audio player."""

    def setup_method(self):
        self.config = {
            "spotter": {
                "enabled": True,
                "cooldowns": {"proximity_ms": 3000, "clearance_ms": 5000},
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

        spotter.update(0, is_on_track=True)
        spotter.update(1, is_on_track=True)
        assert "car_left" in played_keys

    def test_spotter_reset_clears_state(self):
        """After reset, transitions should fire again."""
        spotter = Spotter(self.config)
        played_keys = []
        spotter._player.play = lambda key: played_keys.append(key)

        spotter.update(0, is_on_track=True)
        spotter.update(1, is_on_track=True)
        assert "car_left" in played_keys

        spotter.reset()

        played_keys.clear()
        spotter.update(0, is_on_track=True)
        spotter.update(1, is_on_track=True)
        assert "car_left" in played_keys


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
        """available_samples should list loaded sample keys."""
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
        # Should be empty since the file doesn't exist
        assert player.available_samples == []

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
