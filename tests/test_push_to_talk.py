"""Unit tests for modules/push_to_talk.py — wheel-button hold-to-talk."""

import time

from modules.push_to_talk import PushToTalkController


class FakeSTT:
    """Event-controlled STT: record_until_release waits, transcribe returns canned text."""

    def __init__(self, transcript="box this lap", delay=0.1):
        self.transcript = transcript
        self.delay = delay
        self.record_calls = 0
        self.transcribe_calls = 0
        self.last_max_duration = None

    def record_until_release(self, stop_event, max_duration_s=15):
        self.record_calls += 1
        self.last_max_duration = max_duration_s
        stop_event.wait(timeout=5)
        return b"audio"

    def transcribe(self, audio):
        self.transcribe_calls += 1
        return self.transcript


def _wait_for(predicate, timeout=3.0):
    """Spin until predicate() is truthy or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestPushToTalk:
    def test_press_release_fires_on_question(self):
        """press → release → on_question fires with the transcript + timing."""
        received = []

        def on_question(text, timing_steps):
            received.append((text, timing_steps))

        stt = FakeSTT()
        controller = PushToTalkController(stt, on_question=on_question)

        controller.on_press()
        assert controller._recording.is_set()

        controller.on_release()
        # record_until_release unblocks on stop; transcription runs in thread
        assert _wait_for(lambda: len(received) == 1)
        text, timing_steps = received[0]
        assert text == "box this lap"
        assert any(name == "Recording" for name, _ in timing_steps)
        assert any(name == "Transcription" for name, _ in timing_steps)
        # Recording flag cleared once the query thread finished
        assert not controller._recording.is_set()

    def test_press_while_recording_is_ignored(self):
        """A second press while recording (<30s) is ignored, no second thread."""
        received = []

        def on_question(text, timing_steps):
            received.append(text)

        stt = FakeSTT()
        controller = PushToTalkController(stt, on_question=on_question)

        controller.on_press()
        # Second press while recording — should be ignored
        controller.on_press()
        assert controller._recording.is_set()

        controller.on_release()
        assert _wait_for(lambda: len(received) == 1)
        # Only one recording/transcription happened
        assert stt.record_calls == 1

    def test_max_record_seconds_passed_through(self):
        """max_record_seconds from config reaches record_until_release."""
        received = []

        def on_question(text, timing_steps):
            received.append(text)

        stt = FakeSTT()
        controller = PushToTalkController(
            stt, on_question=on_question, max_record_seconds=7
        )

        controller.on_press()
        controller.on_release()
        assert _wait_for(lambda: len(received) == 1)
        assert stt.last_max_duration == 7

    def test_stuck_recording_force_clears(self):
        """A >30s stuck recording is force-cleared on the next press."""
        received = []

        def on_question(text, timing_steps):
            received.append(text)

        stt = FakeSTT(delay=0)  # record_until_release returns immediately
        controller = PushToTalkController(stt, on_question=on_question)

        # Simulate a stuck recording: set the flag directly with an old timestamp
        controller._recording.set()
        controller._recording_time[0] = time.monotonic() - 31

        # Next press force-clears the stuck recording (generation bumped so the
        # stale thread can't clobber newer state) and starts a fresh one.
        controller.on_press()
        assert controller._generation[0] == 1  # old thread retired
        assert controller._recording.is_set()  # fresh recording started
        assert controller._recording_time[0] > time.monotonic() - 1  # timestamp reset
        assert not controller._stop.is_set()  # stop cleared for the fresh recording
