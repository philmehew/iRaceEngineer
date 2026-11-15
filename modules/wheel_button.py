"""
Wheel button listener — pygame-based steering wheel button input.

Runs a background thread that polls pygame events and calls on_press/on_release
callbacks when any configured button is triggered. Used for both LLM query
triggers (press to query) and push-to-talk voice input (hold to record, release
to transcribe).

Each binding identifies a steering wheel by device name and/or USB GUID, plus a
button index. Device identity is used instead of the joystick's connection
index, which changes when devices are plugged in or removed and is always 0
when only one wheel is connected:

    voice:
      trigger:
        buttons:
          - name: Formula
            device_name: "Simucube 2 Pro"
            device_guid: "030000005e0400008e02000000000000"
            button_index: 9
          - name: GT
            device_name: "FANATEC Podium DD1"
            button_index: 4

If both device_name and device_guid are given, both must match. If only one is
given, that field is used. The listener treats all configured buttons as one
trigger: pressing any button starts, and all must be released to stop.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("iraceengineer")


def _normalize_name(value: Any) -> str:
    """Normalize a device name for case-insensitive comparison."""
    return str(value or "").strip().casefold()


def _clean_guid(value: Any) -> str:
    """Normalize a GUID for storage/display without removing separators."""
    text = str(value or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def _normalize_guid(value: Any) -> str:
    """Normalize a device GUID for comparison.

    SDL reports GUIDs as hex strings. Accept common separators and an optional
    ``0x`` prefix so copied values match regardless of formatting.
    """
    return "".join(ch for ch in _clean_guid(value) if ch not in " -:_")


@dataclass(frozen=True)
class ButtonBinding:
    """One steering-wheel button, matched to a device by name and/or GUID."""

    button_index: int
    device_name: str = ""
    device_guid: str = ""
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "device_name", str(self.device_name or "").strip())
        guid = str(self.device_guid or "").strip().lower()
        if guid.startswith("0x"):
            guid = guid[2:]
        object.__setattr__(self, "device_guid", guid)
        object.__setattr__(self, "name", str(self.name or "").strip())
        if not self.device_name and not self.device_guid:
            raise ValueError(
                "ButtonBinding requires device_name or device_guid "
                "(device_index is not stable across connections)"
            )

    @property
    def device_label(self) -> str:
        """Human-readable device identity for logs and messages."""
        return self.device_name or self.device_guid

    @property
    def label(self) -> str:
        target = f"button {self.button_index} ({self.device_label})"
        return f"{self.name}: {target}" if self.name else target

    def matches(self, joystick: Any) -> bool:
        """Return True when *joystick* is the device this binding targets.

        Both fields must match when both are configured. Raises if the
        joystick's identity cannot be read (the caller treats that as a lost
        device).
        """
        if self.device_guid:
            get_guid = getattr(joystick, "get_guid", None)
            if get_guid is None:
                return False
            if _normalize_guid(get_guid()) != _normalize_guid(self.device_guid):
                return False
        if self.device_name:
            get_name = getattr(joystick, "get_name", None)
            if get_name is None:
                return False
            if _normalize_name(get_name()) != _normalize_name(self.device_name):
                return False
        return True


def _as_int(value: Any) -> int:
    """Coerce a config value to int, rejecting bools and bad strings."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a button index")
    return int(value)


def _dedupe_bindings(bindings: list[ButtonBinding]) -> list[ButtonBinding]:
    """Drop duplicate device/button pairs, preserving order."""
    seen: set[tuple[str, str, int]] = set()
    result: list[ButtonBinding] = []
    for binding in bindings:
        key = (
            _normalize_guid(binding.device_guid),
            _normalize_name(binding.device_name),
            binding.button_index,
        )
        if key not in seen:
            seen.add(key)
            result.append(binding)
    return result


def button_bindings_from_config(trigger_config: dict) -> list[ButtonBinding]:
    """Return the configured wheel button bindings.

    Expects one entry per steering wheel::

        buttons:
          - name: Formula
            device_name: "Simucube 2 Pro"
            device_guid: "030000005e0400008e02000000000000"
            button_index: 9

    ``device_name`` and ``device_guid`` identify the wheel itself, so bindings
    keep working regardless of connection order. At least one is required;
    ``device_index`` is ignored because it is not stable.

    A missing or empty ``buttons`` list means no wheel triggers.
    """
    raw_buttons = trigger_config.get("buttons")
    if raw_buttons is None:
        return []
    if not isinstance(raw_buttons, list):
        logger.warning(
            "voice.trigger.buttons must be a list — no wheel buttons configured"
        )
        return []

    bindings: list[ButtonBinding] = []
    for i, entry in enumerate(raw_buttons):
        if not isinstance(entry, dict):
            logger.warning(f"Ignoring voice.trigger.buttons[{i}]: expected a mapping")
            continue

        raw_name = entry.get("name")
        name = str(raw_name).strip() if raw_name is not None else ""
        raw_device_name = entry.get("device_name")
        device_name = (
            str(raw_device_name).strip() if raw_device_name is not None else ""
        )
        raw_device_guid = entry.get("device_guid")
        device_guid = (
            str(raw_device_guid).strip() if raw_device_guid is not None else ""
        )

        if "device_index" in entry:
            logger.warning(
                f"voice.trigger.buttons[{i}]: device_index is ignored — "
                "match wheels by device_name and/or device_guid instead"
            )

        if not device_name and not device_guid:
            logger.warning(
                f"Ignoring voice.trigger.buttons[{i}]: device_name or "
                "device_guid is required (run scripts/test_wheel.py to find them)"
            )
            continue

        try:
            button_index = _as_int(entry["button_index"])
        except KeyError as e:
            logger.warning(f"Ignoring voice.trigger.buttons[{i}]: missing {e.args[0]}")
            continue
        except (TypeError, ValueError):
            logger.warning(
                f"Ignoring voice.trigger.buttons[{i}]: button_index must be an integer"
            )
            continue

        bindings.append(
            ButtonBinding(
                button_index=button_index,
                device_name=device_name,
                device_guid=device_guid,
                name=name,
            )
        )

    return _dedupe_bindings(bindings)


class WheelButtonListener:
    """Listen for configured buttons across one or more joystick devices.

    Runs a background thread that polls pygame events and calls
    on_press/on_release callbacks when any configured button is triggered.

    Used for both LLM query triggers (press to query) and push-to-talk
    voice input (hold to record, release to transcribe).

    Devices are matched by name/GUID rather than connection index, so a
    binding follows its wheel regardless of plug order. Bindings whose device
    isn't connected are ignored; the listener keeps scanning and starts using
    them when the device appears (e.g. after a wheel swap).
    """

    def __init__(
        self,
        bindings: list[ButtonBinding],
        on_press=None,
        on_release=None,
    ):
        self.bindings = _dedupe_bindings(bindings)
        self.on_press = on_press
        self.on_release = on_release
        self._running = False
        self._thread: threading.Thread | None = None
        self._joysticks: dict[int, Any] = {}
        self._ignored_indices: set[int] = set()
        self._known_count: int | None = None
        self._warned_unavailable: set[ButtonBinding] = set()
        self._pygame_initialized = False
        self._was_pressed = False

    def start(self):
        """Start listening for button events in a background thread."""
        if not self.bindings:
            logger.error(
                "Wheel button listener: no buttons configured — set "
                "voice.trigger.buttons in config.yaml."
            )
            return

        # Skip if already running, but allow restart after the thread died
        if self._running and self._thread and self._thread.is_alive():
            return
        if self._thread and self._thread.is_alive():
            return  # Old thread still winding down — don't start a second

        try:
            import importlib.util

            if not importlib.util.find_spec("pygame"):
                raise ImportError
        except (ImportError, ModuleNotFoundError):
            logger.error("pygame not installed. Install with: uv sync --extra wheel")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening and clean up pygame."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._close_joysticks()
        if self._pygame_initialized:
            try:
                import pygame

                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False

    def _close_joysticks(self):
        """Close every open joystick handle."""
        for joystick in self._joysticks.values():
            try:
                joystick.quit()
            except Exception:
                pass
        self._joysticks.clear()
        self._ignored_indices.clear()

    def _drop_joystick(self, index: int):
        """Close and forget one joystick handle."""
        joystick = self._joysticks.pop(index, None)
        if joystick is not None:
            try:
                joystick.quit()
            except Exception:
                pass

    def _open_matching_devices(self, pygame):
        """Open handles for connected devices that match a configured binding."""
        try:
            count = pygame.joystick.get_count()
        except Exception:
            return

        for index in range(count):
            if index in self._joysticks or index in self._ignored_indices:
                continue
            try:
                joystick = pygame.joystick.Joystick(index)
                joystick.init()
            except Exception as e:
                logger.debug(
                    f"Wheel button listener: device {index} not available ({e})"
                )
                continue

            try:
                matched = any(binding.matches(joystick) for binding in self.bindings)
            except Exception as e:
                logger.debug(
                    f"Wheel button listener: device {index} identity check failed ({e})"
                )
                matched = False

            if not matched:
                try:
                    joystick.quit()
                except Exception:
                    pass
                self._ignored_indices.add(index)
                continue

            self._joysticks[index] = joystick
            try:
                device_name = joystick.get_name()
            except Exception:
                device_name = "unknown"
            logger.info(f"Wheel button listener: opened device {index} ({device_name})")

    def _read_pressed_bindings(self) -> list[ButtonBinding]:
        """Return the bindings whose button is currently held."""
        pressed: list[ButtonBinding] = []
        for binding in self.bindings:
            for index, joystick in list(self._joysticks.items()):
                try:
                    if not binding.matches(joystick):
                        continue
                    num_buttons = joystick.get_numbuttons()
                    if binding.button_index >= num_buttons:
                        if binding not in self._warned_unavailable:
                            self._warned_unavailable.add(binding)
                            logger.warning(
                                f"Configured {binding.label} does not exist on the "
                                f"connected wheel ({num_buttons} buttons) — ignoring"
                            )
                        continue
                    if joystick.get_button(binding.button_index):
                        pressed.append(binding)
                        break
                except Exception as e:
                    logger.warning(f"Wheel device {index} lost ({e}) — will reconnect")
                    self._drop_joystick(index)
                    continue
        return pressed

    def _update_press_state(self, is_pressed: bool, detail: str = ""):
        """Fire on_press/on_release on aggregate button-state edges.

        The configured buttons act as one trigger: on_press fires when any
        button goes down, on_release when the last one comes up.
        """
        if is_pressed and not self._was_pressed:
            self._was_pressed = True
            logger.info(f"Wheel trigger pressed ({detail})")
            if self.on_press:
                try:
                    self.on_press()
                except Exception:
                    logger.exception("Wheel button on_press callback failed")
        elif not is_pressed and self._was_pressed:
            self._was_pressed = False
            logger.info("Wheel trigger released")
            if self.on_release:
                try:
                    self.on_release()
                except Exception:
                    logger.exception("Wheel button on_release callback failed")

    def _run(self):
        """Background thread: init pygame, open devices, poll button state.

        Uses direct get_button() polling to detect press/release edges.
        This approach is more reliable than pygame's event system because:
        - It works in background threads on Windows
        - It doesn't require a display window
        - It doesn't depend on the Windows message pump

        Device add/remove events are still drained so wheel swaps can be
        picked up without restarting the app.
        """
        try:
            import pygame
        except ImportError:
            logger.error("pygame not installed — cannot use wheel button trigger")
            self._running = False
            return

        pygame.init()
        pygame.joystick.init()
        self._pygame_initialized = True
        self._known_count = pygame.joystick.get_count()
        self._open_matching_devices(pygame)

        if self._joysticks:
            logger.info(
                f"Wheel button listener: {len(self.bindings)} binding(s) across "
                f"{len(self._joysticks)} device(s) — open: {sorted(self._joysticks)}"
            )
        else:
            targets = ", ".join(
                sorted({binding.device_label for binding in self.bindings})
            )
            logger.warning(
                f"Wheel button listener: none of the configured devices "
                f"({targets}) are connected yet — waiting"
            )

        last_log_time = 0
        last_scan_time = time.monotonic()

        try:
            while self._running:
                # Drain joystick hot-plug events; polling get_button() still
                # handles the actual button state.
                rescan = False
                try:
                    for event in pygame.event.get():
                        if event.type in (
                            pygame.JOYDEVICEADDED,
                            pygame.JOYDEVICEREMOVED,
                        ):
                            rescan = True
                except Exception as e:
                    logger.debug(f"Wheel button listener: event pump failed ({e})")

                try:
                    count = pygame.joystick.get_count()
                    if count != self._known_count:
                        self._known_count = count
                        rescan = True
                except Exception:
                    pass

                if rescan:
                    logger.info(
                        "Wheel button listener: controller list changed — re-scanning"
                    )
                    self._close_joysticks()
                    self._open_matching_devices(pygame)
                    last_scan_time = time.monotonic()
                elif time.monotonic() - last_scan_time > 1.0:
                    # Retry devices that were missing at startup
                    last_scan_time = time.monotonic()
                    self._open_matching_devices(pygame)

                pressed = self._read_pressed_bindings()
                detail = ", ".join(binding.label for binding in pressed)
                self._update_press_state(bool(pressed), detail)

                # Periodic alive log (every 30s) to confirm thread is running
                now = time.monotonic()
                if now - last_log_time > 30:
                    last_log_time = now
                    logger.debug(
                        f"Wheel button listener alive: "
                        f"{len(self._joysticks)} device(s) open, state="
                        f"{'pressed' if self._was_pressed else 'released'}"
                    )

                time.sleep(1 / 30)  # ~30Hz polling

        except Exception as e:
            logger.exception(f"Wheel button listener error: {e}")
        finally:
            self._close_joysticks()
            pygame.joystick.quit()
            pygame.quit()
            self._pygame_initialized = False
            self._was_pressed = False
            # Allow a later start() call to spin up a fresh listener thread
            self._running = False
