"""Unit tests for modules/engineer_query.py — the shared LLM query seam.

Covers _clean_response quirk handling, the ask_engineer seam (LLM call,
[CMD:] macro strip/fire, QA history, error/empty paths), and the
handle_button_press double-press guard.
"""

import threading
import time

from modules.chat_macros import ChatMacroManager
from modules.engineer_query import (
    _clean_response,
    _llm_call_lock,
    ask_engineer,
    handle_button_press,
)
from modules.llm_client import LLMError


# --- _clean_response ---


class TestCleanResponse:
    def test_tilde_to_about(self):
        assert "about 2" in _clean_response("Fuel ~2 laps")

    def test_unit_abbreviations(self):
        cleaned = _clean_response("At 12 m/s or 100 km/h, tyre pressure 34 PSI")
        assert "meters per second" in cleaned
        assert "kilometers per hour" in cleaned
        assert "P S I" in cleaned

    def test_kph_variant(self):
        assert "kilometers per hour" in _clean_response("Speed 120 kph")

    def test_litres(self):
        assert "12.5 litres" in _clean_response("Fuel 12.5 L")

    def test_single_litre(self):
        assert "1 litre" in _clean_response("1 litres")


# --- ask_engineer ---


class StubLLM:
    """Duck-typed LLM client: .ask(messages) returns canned text."""

    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.called = False

    def ask(self, messages):
        self.called = True
        if self.error:
            raise self.error
        return self.response


class StubState:
    """Duck-typed RaceState: get_snapshot() returns a canned dict."""

    def get_snapshot(self):
        return {"lap": 10, "position": 4}


class StubContextBuilder:
    """Duck-typed ContextBuilder: build_prompt + qa_history."""

    def __init__(self, depth="full"):
        from modules.context_builder import QAHistory

        self.context_depth = depth
        self.qa_history = QAHistory(max_exchanges=3)

    def build_prompt(self, snapshot, question=""):
        self.last_snapshot = snapshot
        return [
            {"role": "system", "content": "you are the race engineer"},
            {"role": "user", "content": question or "What's my strategy?"},
        ]


class TestAskEngineer:
    def test_returns_cleaned_response(self):
        llm = StubLLM(response="Fuel ~2 laps, PSI 34")
        builder = StubContextBuilder()
        result = ask_engineer(StubState(), builder, llm, "Fuel check")
        assert result is not None
        assert "about 2" in result
        assert "P S I" in result
        assert llm.called

    def test_fires_chat_macro_actions(self):
        # Real ChatMacroManager with one action configured
        chat_macros = ChatMacroManager(
            {
                "chat_macros": {
                    "actions": {
                        "clear_black_flag": {
                            "macro": 15,
                            "command": "!clearall",
                            "description": "clear black flag",
                        }
                    }
                }
            }
        )
        assert chat_macros.enabled
        llm = StubLLM(response="Cleared it [CMD:clear_black_flag]")
        builder = StubContextBuilder()
        result = ask_engineer(
            StubState(), builder, llm, "Clear the black flag", chat_macros=chat_macros
        )
        assert result is not None
        assert "[CMD:" not in result  # tag stripped, never spoken
        assert "Cleared it" in result

    def test_appends_qa_history(self):
        llm = StubLLM(response="Pit now")
        builder = StubContextBuilder()
        ask_engineer(StubState(), builder, llm, "Should I pit?")
        assert builder.qa_history.as_messages()[-1]["content"] == "Pit now"

    def test_empty_response_returns_none(self):
        llm = StubLLM(response="")
        builder = StubContextBuilder()
        assert ask_engineer(StubState(), builder, llm, "Q") is None

    def test_llm_error_returns_none(self):
        llm = StubLLM(error=LLMError("LLM call failed"))
        builder = StubContextBuilder()
        assert ask_engineer(StubState(), builder, llm, "Q") is None


# --- handle_button_press ---


class TestHandleButtonPress:
    def test_double_press_guard(self):
        """A second press while an LLM call is in flight is skipped."""
        release = threading.Event()

        class SlowLLM:
            def ask(self, messages):
                release.wait(timeout=5)
                return "done"

        calls = []

        class GuardBuilder(StubContextBuilder):
            def build_prompt(self, snapshot, question=""):
                return [{"role": "user", "content": question or "strategy"}]

        def press():
            handle_button_press(StubState(), GuardBuilder(), SlowLLM())
            calls.append("finished")

        # First press starts an LLM call that blocks until release
        t1 = threading.Thread(target=press)
        t1.start()
        time.sleep(0.1)  # let the first call acquire the lock

        # Second press must be skipped (lock busy) and return immediately
        handle_button_press(StubState(), GuardBuilder(), SlowLLM())

        release.set()
        t1.join(timeout=5)

        # Only one thread ran the full pipeline; the second call was skipped
        assert calls == ["finished"]
        # The first call's finally must have released the lock
        assert not _llm_call_lock.locked()
