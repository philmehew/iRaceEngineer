"""
Wheel button listener — pygame-based steering wheel button input.

Runs a background thread that polls pygame events and calls on_press/on_release
callbacks when any configured button is triggered. Used for both LLM query
triggers (press to query) and push-to-talk voice input (hold to record, release
to transcribe).

Each binding pairs a device index with a button index, so several steering
wheels can be configured independently:

    voice:
      trigger:
        buttons:
          - name: Formula
            device_index: 0
            button_index: 9
          - name: GT
            device_index: 1
            button_index: 4

The listener treats all configured buttons as one trigger: pressing any button
starts, and all must be released to stop.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("iraceengineer")


@dataclass(frozen=True)
class ButtonBinding:
    """One steering-wheel button: a device index plus a button index."""

    device_index: int
    button_index: int
    name: str = ""

    @property
    def label(self) -> str:
        target = f"button {self.button_index} (device {self.device_index})"
        return f"{self.name}: {target}" if self.name else target


def _as_int(value: Any) -> int:
    """Coerce a config value to int, rejecting bools and bad strings."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a button index")
    return int(value)


def _dedupe_bindings(bindings: list[ButtonBinding]) -> list[ButtonBinding]:
    """Drop duplicate device/button pairs, preserving order."""
    seen: set[tuple[int, int]] = set()
    result: list[ButtonBinding] = []
    for binding in bindings:
        key = (binding.device_index, binding.button_index)
        if key not in seen:
            seen.add(key)
            result.append(binding)
    return result


def button_bindings_from_config(trigger_config: dict) -> list[ButtonBinding]:
    """Return the configured wheel button bindings.

    Expects one entry per steering wheel::

        buttons:
          - name: Formula
            device_index: 0
            button_index: 9

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
        try:
            device_index = _as_int(entry["device_index"])
            button_index = _as_int(entry["button_index"])
        except KeyError as e:
            logger.warning(f"Ignoring voice.trigger.buttons[{i}]: missing {e.args[0]}")
            continue
        except (TypeError, ValueError):
            logger.warning(
                f"Ignoring voice.trigger.buttons[{i}]: device_index and "
                "button_index must be integers"
            )
            continue

        bindings.append(ButtonBinding(device_index, button_index, name))

    return _dedupe_bindings(bindings)


class WheelButtonListener:
    """Listen for configured buttons across one or more joystick devices.

    Runs a background thread that polls pygame events and calls
    on_press/on_release callbacks when any configured button is triggered.

    Used for both LLM query triggers (press to query) and push-to-talk
    voice input (hold to record, release to transcribe).

    Bindings whose device isn't connected are ignored; the listener keeps
    scanning and starts using them when the device appears (e.g. after a
    wheel swap).
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
        self._known_count: int | None = None
        self._warned_unavailable: set[tuple[int, int]] = set()
        self._pygame_initialized = False
        self._was_pressed = False

    @property
    def device_indices(self) -> list[int]:
        return sorted({binding.device_index for binding in self.bindings})

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

    def _open_missing_devices(self, pygame):
        """Open handles for configured devices that are now available."""
        try:
            count = pygame.joystick.get_count()
        except Exception:
            return

        for index in self.device_indices:
            if index in self._joysticks or index >= count:
                continue
            try:
                joystick = pygame.joystick.Joystick(index)
                joystick.init()
            except Exception as e:
                logger.debug(
                    f"Wheel button listener: device {index} not available ({e})"
                )
                continue
            self._joysticks[index] = joystick
            logger.info(
                f"Wheel button listener: opened device {index} ({joystick.get_name()})"
            )

    def _read_pressed_bindings(self) -> list[ButtonBinding]:
        """Return the bindings whose button is currently held."""
        pressed: list[ButtonBinding] = []
        for binding in self.bindings:
            joystick = self._joysticks.get(binding.device_index)
            if joystick is None:
                continue
            try:
                num_buttons = joystick.get_numbuttons()
                if binding.button_index >= num_buttons:
                    key = (binding.device_index, binding.button_index)
                    if key not in self._warned_unavailable:
                        self._warned_unavailable.add(key)
                        logger.warning(
                            f"Configured {binding.label} does not exist on the "
                            f"connected wheel ({num_buttons} buttons) — ignoring"
                        )
                    continue
                if joystick.get_button(binding.button_index):
                    pressed.append(binding)
            except Exception as e:
                logger.warning(
                    f"Wheel device {binding.device_index} lost ({e}) — will reconnect"
                )
                joystick = self._joysticks.pop(binding.device_index, None)
                if joystick is not None:
                    try:
                        joystick.quit()
                    except Exception:
                        pass
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
        self._open_missing_devices(pygame)

        if self._joysticks:
            logger.info(
                f"Wheel button listener: {len(self.bindings)} binding(s) across "
                f"{self.device_indices} — open device(s) {sorted(self._joysticks)}"
            )
        else:
            logger.warning(
                f"Wheel button listener: none of the configured devices "
                f"{self.device_indices} are connected yet — waiting"
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
                    self._open_missing_devices(pygame)
                    last_scan_time = time.monotonic()
                elif time.monotonic() - last_scan_time > 1.0:
                    # Retry devices that were missing at startup
                    last_scan_time = time.monotonic()
                    self._open_missing_devices(pygame)

                pressed = self._read_pressed_bindings()
                detail = ", ".join(binding.label for binding in pressed)
                self._update_press_state(bool(pressed), detail)

                # Periodic alive log (every 30s) to confirm thread is running
                now = time.monotonic()
                if now - last_log_time > 30:
                    last_log_time = now
                    logger.debug(
                        f"Wheel button listener alive: "
                        f"{len(self._joysticks)}/{len(self.device_indices)} "
                        f"device(s) open, state="
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
