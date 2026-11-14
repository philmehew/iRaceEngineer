"""
Wheel button listener — pygame-based steering wheel button input.

Runs a background thread that polls pygame events and calls on_press/on_release
callbacks when the target button is triggered. Used for both LLM query triggers
(press to query) and push-to-talk voice input (hold to record, release to
transcribe).
"""

import logging
import threading
import time

logger = logging.getLogger("iraceengineer")


class WheelButtonListener:
    """Listen for joystick/wheel button presses using pygame.

    Runs a background thread that polls pygame events and calls
    on_press/on_release callbacks when the target button is triggered.

    Used for both LLM query triggers (press to query) and push-to-talk
    voice input (hold to record, release to transcribe).

    Config keys:
        device_index: Joystick device index (use scripts/test_wheel.py to find)
        button_index: Button index on the device
    """

    def __init__(
        self,
        device_index: int,
        button_index: int,
        on_press=None,
        on_release=None,
    ):
        self.device_index = device_index
        self.button_index = button_index
        self.on_press = on_press
        self.on_release = on_release
        self._running = False
        self._thread: threading.Thread | None = None
        self._joystick = None
        self._pygame_initialized = False

    def start(self):
        """Start listening for button events in a background thread."""
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
        if self._pygame_initialized:
            try:
                import pygame

                if self._joystick is not None:
                    self._joystick.quit()
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False

    def _run(self):
        """Background thread: init pygame, open joystick, poll button state.

        Uses direct get_button() polling to detect press/release edges.
        This approach is more reliable than pygame's event system because:
        - It works in background threads on Windows
        - It doesn't require a display window
        - It doesn't depend on the Windows message pump

        Falls back to event-based detection if polling reports stale state
        (all buttons permanently pressed), which can happen if the event pump
        isn't running.
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

        count = pygame.joystick.get_count()
        logger.info(f"Wheel button listener: pygame init, {count} joystick(s) found")

        if self.device_index >= count:
            logger.error(
                f"Joystick device index {self.device_index} not found "
                f"(only {count} device(s) available). "
                f"Run scripts/test_wheel.py --list to see devices."
            )
            self._running = False
            pygame.quit()
            self._pygame_initialized = False
            return

        self._joystick = pygame.joystick.Joystick(self.device_index)
        self._joystick.init()
        js_name = self._joystick.get_name()
        js_buttons = self._joystick.get_numbuttons()
        logger.info(
            f"Wheel button listener: device {self.device_index} "
            f"({js_name}), button {self.button_index} of {js_buttons}"
        )

        # Track button state to detect press/release edges
        was_pressed = False
        last_log_time = 0

        try:
            while self._running:
                # Pump pygame events so joystick state stays fresh.
                # Even though we read state via get_button(), the internal
                # joystick state only updates when the event pump runs.
                pygame.event.pump()

                if self._joystick is None:
                    # Joystick was disconnected — try to reopen it
                    try:
                        self._joystick = pygame.joystick.Joystick(self.device_index)
                        self._joystick.init()
                        js_name = self._joystick.get_name()
                        logger.info(
                            f"Wheel button listener: reconnected to "
                            f"device {self.device_index} ({js_name})"
                        )
                    except Exception:
                        # Still gone — retry later (don't spam the log)
                        time.sleep(1.0)
                        continue

                try:
                    is_pressed = bool(self._joystick.get_button(self.button_index))
                except Exception as e:
                    # Joystick disconnected mid-read — drop it and retry
                    logger.warning(f"Wheel device lost ({e}) — attempting reconnect")
                    try:
                        self._joystick.quit()
                    except Exception:
                        pass
                    self._joystick = None
                    was_pressed = False  # don't hold a phantom press
                    continue

                if is_pressed and not was_pressed:
                    was_pressed = True
                    logger.info(
                        f"Wheel button {self.button_index} pressed "
                        f"(device {self.device_index}: {js_name})"
                    )
                    if self.on_press:
                        try:
                            self.on_press()
                        except Exception:
                            logger.exception("Wheel button on_press callback failed")
                elif not is_pressed and was_pressed:
                    was_pressed = False
                    logger.info(
                        f"Wheel button {self.button_index} released "
                        f"(device {self.device_index}: {js_name})"
                    )
                    if self.on_release:
                        try:
                            self.on_release()
                        except Exception:
                            logger.exception("Wheel button on_release callback failed")

                # Periodic alive log (every 30s) to confirm thread is running
                now = time.monotonic()
                if now - last_log_time > 30:
                    last_log_time = now
                    logger.debug(
                        f"Wheel button listener alive: device {self.device_index}, "
                        f"button {self.button_index}, state={'pressed' if was_pressed else 'released'}"
                    )

                time.sleep(1 / 30)  # ~30Hz polling

        except Exception as e:
            logger.exception(f"Wheel button listener error: {e}")
        finally:
            if self._joystick is not None:
                try:
                    self._joystick.quit()
                except Exception:
                    pass
            pygame.joystick.quit()
            pygame.quit()
            self._pygame_initialized = False
            # Allow a later start() call to spin up a fresh listener thread
            self._running = False
