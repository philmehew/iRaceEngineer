"""Unit tests for modules/wheel_button.py — multi-wheel button bindings."""

from modules.wheel_button import (
    ButtonBinding,
    WheelButtonListener,
    button_bindings_from_config,
)


class TestButtonBindingsFromConfig:
    def test_list_with_names(self):
        config = {
            "buttons": [
                {"name": "Formula", "device_index": 0, "button_index": 9},
                {"name": "GT", "device_index": 1, "button_index": 4},
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(0, 9, "Formula"),
            ButtonBinding(1, 4, "GT"),
        ]

    def test_no_buttons_returns_empty(self):
        assert button_bindings_from_config({}) == []

    def test_invalid_entries_skipped(self):
        config = {
            "buttons": [
                {"device_index": 0, "button_index": 9},
                "not-a-mapping",
                {"device_index": 1},  # missing button_index
                {"device_index": "x", "button_index": 4},
                {"device_index": 2, "button_index": 7},
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(0, 9),
            ButtonBinding(2, 7),
        ]

    def test_duplicates_removed_order_preserved(self):
        config = {
            "buttons": [
                {"device_index": 0, "button_index": 9},
                {"device_index": 0, "button_index": 4},
                {"device_index": 0, "button_index": 9},
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(0, 9),
            ButtonBinding(0, 4),
        ]

    def test_empty_buttons_returns_empty(self):
        assert button_bindings_from_config({"buttons": []}) == []

    def test_invalid_buttons_container_returns_empty(self):
        assert button_bindings_from_config({"buttons": "nope"}) == []

    def test_string_numbers_coerced(self):
        config = {
            "buttons": [{"device_index": "0", "button_index": "9"}],
        }
        assert button_bindings_from_config(config) == [ButtonBinding(0, 9)]


class TestBindingLabel:
    def test_named_binding(self):
        assert ButtonBinding(0, 9, "Formula").label == "Formula: button 9 (device 0)"

    def test_unnamed_binding(self):
        assert ButtonBinding(1, 4).label == "button 4 (device 1)"


class FakeJoystick:
    def __init__(self, buttons):
        self.buttons = list(buttons)
        self.quit_called = False

    def get_numbuttons(self):
        return len(self.buttons)

    def get_button(self, index):
        return self.buttons[index]

    def quit(self):
        self.quit_called = True


class BrokenJoystick(FakeJoystick):
    def get_numbuttons(self):
        raise RuntimeError("device lost")


class TestReadPressedBindings:
    def test_reads_buttons_from_multiple_devices(self):
        listener = WheelButtonListener(
            [
                ButtonBinding(0, 2, "Formula"),
                ButtonBinding(1, 0, "GT"),
            ]
        )
        listener._joysticks = {
            0: FakeJoystick([False, False, True] + [False] * 7),
            1: FakeJoystick([True] + [False] * 7),
        }

        assert listener._read_pressed_bindings() == [
            ButtonBinding(0, 2, "Formula"),
            ButtonBinding(1, 0, "GT"),
        ]

    def test_missing_device_is_ignored(self):
        listener = WheelButtonListener([ButtonBinding(1, 0)])
        listener._joysticks = {}

        assert listener._read_pressed_bindings() == []

    def test_button_missing_from_connected_wheel_is_ignored(self):
        listener = WheelButtonListener([ButtonBinding(0, 9, "Other wheel")])
        listener._joysticks = {0: FakeJoystick([False] * 4)}

        assert listener._read_pressed_bindings() == []
        assert (0, 9) in listener._warned_unavailable

    def test_lost_device_is_dropped(self):
        listener = WheelButtonListener([ButtonBinding(0, 2)])
        joystick = BrokenJoystick([False] * 4)
        listener._joysticks = {0: joystick}

        assert listener._read_pressed_bindings() == []
        assert listener._joysticks == {}
        assert joystick.quit_called


class TestWheelButtonListenerEdges:
    def _listener(self):
        events = []
        listener = WheelButtonListener(
            [
                ButtonBinding(0, 9, "Formula"),
                ButtonBinding(1, 4, "GT"),
            ],
            on_press=lambda: events.append("press"),
            on_release=lambda: events.append("release"),
        )
        return listener, events

    def test_any_button_starts_once(self):
        listener, events = self._listener()

        listener._update_press_state(True, "Formula")
        listener._update_press_state(True, "Formula, GT")

        assert events == ["press"]

    def test_release_waits_for_all_buttons(self):
        listener, events = self._listener()

        listener._update_press_state(True, "Formula")
        listener._update_press_state(True, "Formula, GT")  # second also held
        listener._update_press_state(True, "GT")  # one released, one held
        listener._update_press_state(False, "")  # last released

        assert events == ["press", "release"]

    def test_release_without_press_is_ignored(self):
        listener, events = self._listener()

        listener._update_press_state(False, "")

        assert events == []

    def test_new_press_after_release_fires_again(self):
        listener, events = self._listener()

        listener._update_press_state(True, "Formula")
        listener._update_press_state(False, "")
        listener._update_press_state(True, "GT")

        assert events == ["press", "release", "press"]
