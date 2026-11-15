"""
Standalone wheel/controller button discovery — find a device's identity and
button index for push-to-talk on a steering wheel or game controller.

Lists connected joystick devices and watches for button presses, printing the
exact config.yaml entries you need for wheel_button trigger mode.

Wheels are matched by device name and/or USB GUID rather than connection
index, because the index changes when devices are plugged in or removed and is
always 0 when only one wheel is connected.

IMPORTANT: pygame assigns each joystick an "instance_id" that may differ from
the device index. This script maps events to the correct device using the
instance_id, so the config values it suggests are always correct.

Usage:
    python scripts/test_wheel.py --list              # List connected devices
    python scripts/test_wheel.py                      # Watch for button presses
    python scripts/test_wheel.py --device 0           # Watch only device 0
    python scripts/test_wheel.py --watch-axes         # Also show analog axis movements
"""

import argparse
import json
import sys
import time

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _device_name(js) -> str:
    """Return a joystick's name, or an empty string if it can't be read."""
    try:
        return str(js.get_name())
    except Exception:
        return ""


def _device_guid(js) -> str:
    """Return a joystick's USB GUID, or an empty string if unavailable."""
    try:
        return str(js.get_guid())
    except Exception:
        return ""


def list_devices():
    """List all connected joystick/controller devices."""
    try:
        import pygame
    except ImportError:
        print(
            "ERROR: pygame not installed. Run: uv sync --extra wheel\n"
            "       or: pip install pygame>=2.6.0"
        )
        return

    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        print("\n🎮 No joystick/controller devices found.")
        print("   Make sure your wheel is connected and drivers are installed.")
        pygame.quit()
        return

    print(f"\n🎮 Found {count} device(s):\n")
    print(
        f"  {'Idx':<4} {'Instance':<9} {'Name':<40} {'GUID':<34} "
        f"{'Btns':<5} {'Axes':<5} {'Hats'}"
    )
    print("  " + "-" * 105)

    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        iid = js.get_instance_id() if hasattr(js, "get_instance_id") else i
        print(
            f"  {i:<4} {iid:<9} {_device_name(js):<40} {_device_guid(js):<34} "
            f"{js.get_numbuttons():<5} {js.get_numaxes():<5} {js.get_numhats()}"
        )
        js.quit()

    print("\n  'Idx' is only the current connection order — it changes when")
    print("  devices are plugged/unplugged. In config.yaml, match wheels by")
    print("  device_name and/or device_guid instead.")
    print("  Run without --list to watch for button presses.\n")
    pygame.quit()


def watch_buttons(device_index: int | None = None, watch_axes: bool = False):
    """Watch for button presses on joystick devices.

    Args:
        device_index: Specific device index to monitor, or None for all devices.
            This is only for monitoring; config uses device_name/device_guid.
        watch_axes: If True, also print analog axis movements.
    """
    try:
        import pygame
    except ImportError:
        print(
            "ERROR: pygame not installed. Run: uv sync --extra wheel\n"
            "       or: pip install pygame>=2.6.0"
        )
        return

    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        print("\n🎮 No joystick/controller devices found.")
        print("   Make sure your wheel is connected and drivers are installed.\n")
        pygame.quit()
        return

    # Open joystick(s) and build instance_id → (device_index, joystick) mapping.
    # pygame's event.instance_id may differ from the device index used to open
    # the joystick, so we need this mapping to identify which physical device
    # an event came from.
    joysticks = {}  # device_index → Joystick object
    instance_map = {}  # instance_id → device_index
    open_indices = []

    if device_index is not None:
        if device_index >= count:
            print(
                f"\n❌ Device index {device_index} not found. "
                f"Available: 0-{count - 1}\n"
            )
            pygame.quit()
            return
        indices = [device_index]
    else:
        indices = list(range(count))

    print(f"\n🎮 Monitoring {len(indices)} device(s) for button presses...\n")

    for i in indices:
        js = pygame.joystick.Joystick(i)
        js.init()
        iid = js.get_instance_id() if hasattr(js, "get_instance_id") else i
        joysticks[i] = js
        instance_map[iid] = i
        open_indices.append(i)
        btns = js.get_numbuttons()
        axes = js.get_numaxes()
        hats = js.get_numhats()
        print(
            f"  Device {i} (instance {iid}): {_device_name(js)} "
            f"({btns} buttons, {axes} axes, {hats} hats)"
        )
        guid = _device_guid(js)
        if guid:
            print(f"    GUID: {guid}")

    print("\n  🎯 Press buttons on your wheel to identify them.")
    print(
        "     The config suggestion uses device_name/device_guid, so the "
        "binding follows the wheel regardless of connection order."
    )
    print("     Press Ctrl+C to exit.\n")

    # Track button/hat states to detect changes
    button_states = {}
    hat_states = {}
    for i in open_indices:
        js = joysticks[i]
        button_states[i] = [False] * js.get_numbuttons()
        hat_states[i] = [(0, 0)] * js.get_numhats()

    # Track last suggested button to avoid repeating the same config snippet
    last_suggestion = None
    # Bindings seen so far, so the suggested list grows as you press the
    # race-engineer button on each of your wheels.
    seen_bindings: list[tuple[str, str, int]] = []

    try:
        while True:
            for event in pygame.event.get():
                # Only handle joystick events — JOYDEVICEADDED/REMOVED and
                # other events carry different attributes (device_index, not
                # instance_id) and would crash the watcher on hot-plug
                if event.type not in (
                    pygame.JOYBUTTONDOWN,
                    pygame.JOYBUTTONUP,
                    pygame.JOYHATMOTION,
                    pygame.JOYAXISMOTION,
                ):
                    if event.type == pygame.JOYDEVICEADDED:
                        print(
                            f"\n  🔌 Device added (index {getattr(event, 'device_index', '?')}) "
                            f"— restart test_wheel.py to see it.\n"
                        )
                    elif event.type == pygame.JOYDEVICEREMOVED:
                        print("\n  ⚠️  Device removed — restart test_wheel.py.\n")
                    continue

                # Resolve the event's instance_id to our device index
                eid = event.instance_id
                dev_idx = instance_map.get(eid)
                if dev_idx is None:
                    # Event from a device we didn't open — skip
                    continue
                name = _device_name(joysticks[dev_idx])

                if event.type == pygame.JOYBUTTONDOWN:
                    btn = event.button
                    print(f"  ✅ Button {btn} PRESSED on device {dev_idx} ({name})")
                    button_states[dev_idx][btn] = True

                    # Print config suggestion keyed by device identity, not
                    # connection index.
                    dev_name = _device_name(joysticks[dev_idx])
                    dev_guid = _device_guid(joysticks[dev_idx])
                    suggestion_key = (dev_name, dev_guid, btn)
                    if suggestion_key != last_suggestion:
                        last_suggestion = suggestion_key
                        if suggestion_key not in seen_bindings:
                            seen_bindings.append(suggestion_key)
                        print("\n  📋 Suggested config.yaml:")
                        print("     voice:")
                        print("       trigger:")
                        print("         buttons:")
                        for index, (d_name, d_guid, button) in enumerate(
                            seen_bindings, start=1
                        ):
                            print(f"           - name: Wheel {index}")
                            if d_name:
                                print(f"             device_name: {json.dumps(d_name)}")
                            if d_guid:
                                print(f"             device_guid: {json.dumps(d_guid)}")
                            if not d_name and not d_guid:
                                print(
                                    "             # device identity unavailable — "
                                    "restart the script"
                                )
                            print(f"             button_index: {button}")
                        print("         max_record_seconds: 15")
                        print(
                            "     (Press your race-engineer button on each wheel "
                            "to add it to the list.)\n"
                        )

                elif event.type == pygame.JOYBUTTONUP:
                    btn = event.button
                    print(f"  🔴 Button {btn} RELEASED on device {dev_idx} ({name})")
                    button_states[dev_idx][btn] = False

                elif event.type == pygame.JOYHATMOTION:
                    hat = event.hat
                    value = event.value
                    old = hat_states[dev_idx][hat]
                    if value != old:
                        direction = _hat_direction(value)
                        print(
                            f"  🎮 Hat {hat}: {old} → {value} ({direction}) "
                            f"on device {dev_idx} ({name})"
                        )
                        hat_states[dev_idx][hat] = value

                elif event.type == pygame.JOYAXISMOTION and watch_axes:
                    axis = event.axis
                    value = event.value
                    # Only print significant movement (> 10% change from center)
                    if abs(value) > 0.1:
                        print(
                            f"  📊 Axis {axis}: {value:+.3f} "
                            f"on device {dev_idx} ({name})"
                        )

            time.sleep(1 / 60)  # ~60Hz polling, light on CPU

    except KeyboardInterrupt:
        print("\n\n👋 Stopped watching.")
    finally:
        for js in joysticks.values():
            js.quit()
        pygame.quit()


def _hat_direction(value: tuple) -> str:
    """Convert hat value to human-readable direction."""
    x, y = value
    directions = []
    if y > 0:
        directions.append("Up")
    elif y < 0:
        directions.append("Down")
    if x > 0:
        directions.append("Right")
    elif x < 0:
        directions.append("Left")
    return "-".join(directions) if directions else "Center"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone wheel/controller button discovery — "
            "find a device's identity and button index for push-to-talk."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List connected joystick/controller devices and exit",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help=(
            "Monitor only this device index (default: monitor all devices). "
            "This is only for monitoring; config uses device_name/device_guid."
        ),
    )
    parser.add_argument(
        "--watch-axes",
        action="store_true",
        help="Also show analog axis movements (noisy, off by default)",
    )

    args = parser.parse_args()

    if args.list:
        list_devices()
    else:
        watch_buttons(device_index=args.device, watch_axes=args.watch_axes)


if __name__ == "__main__":
    main()
