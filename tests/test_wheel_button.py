"""Unit tests for modules/wheel_button.py — multi-wheel button bindings."""

import pytest

from modules.wheel_button import (
    ButtonBinding,
    WheelButtonListener,
    button_bindings_from_config,
)


class TestButtonBindingsFromConfig:
    def test_list_with_identities(self):
        config = {
            "buttons": [
                {
                    "name": "Formula",
                    "device_name": "Simucube 2 Pro",
                    "device_guid": "guid-a",
                    "button_index": 9,
                },
                {
                    "name": "GT",
                    "device_name": "FANATEC Podium DD1",
                    "button_index": 4,
                },
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(9, "Simucube 2 Pro", "guid-a", "Formula"),
            ButtonBinding(4, "FANATEC Podium DD1", "", "GT"),
        ]

    def test_no_buttons_returns_empty(self):
        assert button_bindings_from_config({}) == []

    def test_invalid_entries_skipped(self):
        config = {
            "buttons": [
                {"device_name": "Wheel A", "button_index": 9},
                "not-a-mapping",
                {"device_name": "Wheel B"},  # missing button_index
                {"device_name": "Wheel C", "button_index": "x"},
                {"device_name": "Wheel D", "button_index": 7},
                {"device_index": 0, "button_index": 4},  # no identity
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(9, "Wheel A"),
            ButtonBinding(7, "Wheel D"),
        ]

    def test_device_index_is_ignored_when_identity_present(self):
        config = {
            "buttons": [
                {"device_name": "Wheel A", "device_index": 0, "button_index": 9},
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(9, "Wheel A"),
        ]

    def test_duplicates_removed_order_preserved(self):
        config = {
            "buttons": [
                {"device_name": "Wheel A", "button_index": 9},
                {"device_name": "Wheel A", "button_index": 4},
                {"device_name": "Wheel A", "button_index": 9},
            ]
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(9, "Wheel A"),
            ButtonBinding(4, "Wheel A"),
        ]

    def test_empty_buttons_returns_empty(self):
        assert button_bindings_from_config({"buttons": []}) == []

    def test_invalid_buttons_container_returns_empty(self):
        assert button_bindings_from_config({"buttons": "nope"}) == []

    def test_string_numbers_coerced(self):
        config = {
            "buttons": [{"device_name": "Wheel A", "button_index": "9"}],
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(9, "Wheel A"),
        ]

    def test_guid_only_binding(self):
        config = {
            "buttons": [{"device_guid": "GUID-A", "button_index": 4}],
        }
        assert button_bindings_from_config(config) == [
            ButtonBinding(4, device_guid="guid-a"),
        ]


class TestBindingLabel:
    def test_named_binding_uses_device_name(self):
        binding = ButtonBinding(9, "Simucube 2 Pro", "guid-a", "Formula")
        assert binding.label == "Formula: button 9 (Simucube 2 Pro)"

    def test_unnamed_binding(self):
        assert ButtonBinding(4, "FANATEC Podium DD1").label == (
            "button 4 (FANATEC Podium DD1)"
        )

    def test_guid_only_binding(self):
        assert ButtonBinding(4, device_guid="guidb").label == "button 4 (guidb)"

    def test_identity_is_required(self):
        with pytest.raises(ValueError):
            ButtonBinding(9)


class FakeJoystick:
    def __init__(self, buttons, name="Wheel A", guid="guid-a"):
        self.buttons = list(buttons)
        self.name = name
        self.guid = guid
        self.quit_called = False

    def init(self):
        return None

    def get_name(self):
        return self.name

    def get_guid(self):
        return self.guid

    def get_numbuttons(self):
        return len(self.buttons)

    def get_button(self, index):
        return self.buttons[index]

    def quit(self):
        self.quit_called = True


class BrokenJoystick(FakeJoystick):
    def get_numbuttons(self):
        raise RuntimeError("device lost")


class DeadJoystick(FakeJoystick):
    def get_name(self):
        raise RuntimeError("device lost")


class FakePygame:
    """Minimal pygame.joystick facade for _open_matching_devices tests."""

    def __init__(self, devices):
        self._devices = devices
        self.joystick = self

    def get_count(self):
        return len(self._devices)

    def Joystick(self, index):
        return self._devices[index]


class TestBindingMatches:
    def test_matches_by_name_only(self):
        binding = ButtonBinding(9, device_name="Wheel A")
        assert binding.matches(FakeJoystick([], name="wheel a", guid="other"))
        assert not binding.matches(FakeJoystick([], name="Wheel B", guid="guid-a"))

    def test_matches_by_guid_only(self):
        binding = ButtonBinding(9, device_guid="GUID-A")
        assert binding.matches(FakeJoystick([], name="Anything", guid="guid-a"))
        assert not binding.matches(FakeJoystick([], name="Wheel A", guid="guid-b"))

    def test_both_must_match(self):
        binding = ButtonBinding(9, "Wheel A", "guid-a")
        assert binding.matches(FakeJoystick([], name="Wheel A", guid="guid-a"))
        assert not binding.matches(FakeJoystick([], name="Wheel A", guid="guid-b"))
        assert not binding.matches(FakeJoystick([], name="Wheel B", guid="guid-a"))

    def test_guid_formatting_is_normalized(self):
        binding = ButtonBinding(9, device_guid="0xAB-CD:EF")
        assert binding.matches(FakeJoystick([], guid="abcdef"))


class TestReadPressedBindings:
    def test_reads_buttons_from_multiple_devices(self):
        listener = WheelButtonListener(
            [
                ButtonBinding(2, "Wheel A", "guid-a", "Formula"),
                ButtonBinding(0, "Wheel B", "guid-b", "GT"),
            ]
        )
        # Device order is deliberately reversed relative to the binding order.
        listener._joysticks = {
            0: FakeJoystick([True] + [False] * 7, name="Wheel B", guid="guid-b"),
            1: FakeJoystick(
                [False, False, True] + [False] * 7, name="Wheel A", guid="guid-a"
            ),
        }

        assert listener._read_pressed_bindings() == [
            ButtonBinding(2, "Wheel A", "guid-a", "Formula"),
            ButtonBinding(0, "Wheel B", "guid-b", "GT"),
        ]

    def test_missing_device_is_ignored(self):
        listener = WheelButtonListener([ButtonBinding(0, "Wheel A")])
        listener._joysticks = {}

        assert listener._read_pressed_bindings() == []

    def test_wrong_wheel_at_same_index_does_not_trigger(self):
        listener = WheelButtonListener([ButtonBinding(2, "Wheel A", "guid-a")])
        listener._joysticks = {
            0: FakeJoystick([False, False, True], name="Wheel B", guid="guid-b"),
        }

        assert listener._read_pressed_bindings() == []

    def test_button_missing_from_connected_wheel_is_ignored(self):
        binding = ButtonBinding(9, "Wheel A", "guid-a", "Other wheel")
        listener = WheelButtonListener([binding])
        listener._joysticks = {
            0: FakeJoystick([False] * 4, name="Wheel A", guid="guid-a"),
        }

        assert listener._read_pressed_bindings() == []
        assert binding in listener._warned_unavailable

    def test_lost_device_is_dropped(self):
        listener = WheelButtonListener([ButtonBinding(2, "Wheel A")])
        joystick = BrokenJoystick([False] * 4, name="Wheel A", guid="guid-a")
        listener._joysticks = {0: joystick}

        assert listener._read_pressed_bindings() == []
        assert listener._joysticks == {}
        assert joystick.quit_called

    def test_dead_identity_is_dropped(self):
        listener = WheelButtonListener([ButtonBinding(2, "Wheel A")])
        joystick = DeadJoystick([False] * 4, name="Wheel A", guid="guid-a")
        listener._joysticks = {0: joystick}

        assert listener._read_pressed_bindings() == []
        assert listener._joysticks == {}
        assert joystick.quit_called


class TestOpenMatchingDevices:
    def test_opens_only_matching_devices(self):
        other = FakeJoystick([False] * 4, name="Wheel B", guid="guid-b")
        matching = FakeJoystick([False] * 4, name="Wheel A", guid="guid-a")
        listener = WheelButtonListener([ButtonBinding(2, "Wheel A", "guid-a")])

        listener._open_matching_devices(FakePygame([other, matching]))

        assert listener._joysticks == {1: matching}
        assert listener._ignored_indices == {0}
        assert other.quit_called
        assert not matching.quit_called

    def test_ignored_devices_are_not_reopened(self):
        other = FakeJoystick([False] * 4, name="Wheel B", guid="guid-b")
        listener = WheelButtonListener([ButtonBinding(2, "Wheel A", "guid-a")])
        pygame = FakePygame([other])

        listener._open_matching_devices(pygame)
        listener._open_matching_devices(pygame)

        assert listener._joysticks == {}
        assert listener._ignored_indices == {0}


class TestWheelButtonListenerEdges:
    def _listener(self):
        events = []
        listener = WheelButtonListener(
            [
                ButtonBinding(9, "Wheel A", "guid-a", "Formula"),
                ButtonBinding(4, "Wheel B", "guid-b", "GT"),
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
