"""
Standalone wheel/controller button discovery — find device and button indices
for push-to-talk on a steering wheel or game controller.

Lists connected joystick devices and watches for button presses, printing
the exact config.yaml entries you need for wheel_button trigger mode.

IMPORTANT: pygame assigns each joystick an "instance_id" that may differ from
the device index. This script maps events to the correct device using the
instance_id, so the config values it suggests are always correct.

Usage:
    python test_wheel.py --list              # List connected devices
    python test_wheel.py                      # Watch for button presses
    python test_wheel.py --device 0           # Watch only device 0
    python test_wheel.py --watch-axes         # Also show analog axis movements
"""

import argparse
import sys
import time

# Fix Windows console encoding for emoji/special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


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
        f"  {'Idx':<5} {'Instance':<10} {'Name':<45} {'Btns':<6} {'Axes':<6} {'Hats'}"
    )
    print("  " + "-" * 80)

    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        iid = js.get_instance_id() if hasattr(js, "get_instance_id") else i
        print(
            f"  {i:<5} {iid:<10} {js.get_name():<45} {js.get_numbuttons():<6} "
            f"{js.get_numaxes():<6} {js.get_numhats()}"
        )
        js.quit()

    print("\n  The 'Idx' column is the device_index for config.yaml.")
    print("  Run without --list to watch for button presses.\n")
    pygame.quit()


def watch_buttons(device_index: int | None = None, watch_axes: bool = False):
    """Watch for button presses on joystick devices.

    Args:
        device_index: Specific device to monitor, or None for all devices.
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
            f"  Device {i} (instance {iid}): {js.get_name()} "
            f"({btns} buttons, {axes} axes, {hats} hats)"
        )

    print("\n  🎯 Press buttons on your wheel to identify them.")
    print(
        "     The 'device_index' in the config suggestion is the correct value to use."
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

    try:
        while True:
            for event in pygame.event.get():
                # Resolve the event's instance_id to our device index
                eid = event.instance_id
                dev_idx = instance_map.get(eid)
                if dev_idx is None:
                    # Event from a device we didn't open — skip
                    continue
                name = joysticks[dev_idx].get_name()

                if event.type == pygame.JOYBUTTONDOWN:
                    btn = event.button
                    print(f"  ✅ Button {btn} PRESSED on device {dev_idx} ({name})")
                    button_states[dev_idx][btn] = True

                    # Print config suggestion
                    suggestion_key = (dev_idx, btn)
                    if suggestion_key != last_suggestion:
                        last_suggestion = suggestion_key
                        print("\n  📋 Suggested config.yaml:")
                        print("     voice:")
                        print("       trigger:")
                        print("         push_to_talk: true")
                        print('         method: "wheel_button"')
                        print(f"         device_index: {dev_idx}")
                        print(f"         button_index: {btn}")
                        print("         max_record_seconds: 15\n")

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
            "find device and button indices for push-to-talk."
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
        help="Monitor only this device index (default: monitor all devices)",
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
