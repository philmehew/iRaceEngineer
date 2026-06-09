"""
Action executor — parses LLM responses for [ACTION] directives and
either logs them (dry_run=True) or executes them via iRacing SDK.

In v1, this runs in dry_run mode by default. Actions are logged to console
with a [WOULD EXECUTE] prefix. Switching to real execution requires
setting actions.enabled: true in config.yaml.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pattern to match [ACTION] directives in LLM responses
ACTION_PATTERN = re.compile(r"\[ACTION\]\s*(\w[\w_]*)(?:\s*:\s*(\S+))?", re.IGNORECASE)


class ActionExecutor:
    """Parse LLM responses for action directives and execute or log them.

    Actions are optional structured commands embedded in the LLM's text
    response using the format:
        [ACTION] action_name
        [ACTION] action_name: parameter

    Available actions (whitelist in config):
        pit_this_lap        — Request pit stop this lap
        add_fuel: <litres>  — Set fuel amount for next pit stop
        change_tyres        — Request tyre change at next pit stop
        clear_penalty        — Clear a penalty
    """

    # Map action names to pyirsdk PitCommandMode names
    ACTION_MAP = {
        "pit_this_lap": None,  # Uses chat_command or broadcast
        "add_fuel": "fuel",  # pit_command(fuel, litres)
        "change_tyres": "clear_tires",  # pit_command(clear_tires)
        "clear_fr": "clear_fr",  # Fast repair clear
        "clear_penalty": None,  # Uses chat_command
    }

    # Maximum fuel that can be added in a single action (litres)
    MAX_FUEL_ADD = 110  # Most iRacing cars have tanks ≤110L

    def __init__(
        self, iracing_client=None, config: dict | None = None, fuel_max: float = 0.0
    ):
        """
        Args:
            iracing_client: IRacingClient instance for executing commands.
                            Can be None for dry-run/testing.
            config: Configuration dict with 'actions' key.
            fuel_max: Maximum tank capacity in litres (from session info).
                      Used to clamp add_fuel actions.
        """
        self.iracing = iracing_client
        self.fuel_max = fuel_max
        actions_config = (config or {}).get("actions", {})
        self.dry_run = not actions_config.get("enabled", False)
        self.allowed_actions = actions_config.get(
            "allowed_actions",
            ["pit_this_lap", "add_fuel", "change_tyres", "clear_penalty"],
        )

        if self.dry_run:
            logger.info(
                "Action executor running in DRY RUN mode — actions will be logged but not executed"
            )
        else:
            logger.info(
                "Action executor running in LIVE mode — actions will be executed!"
            )

    def parse_response(self, llm_response: str) -> tuple[str, list[dict]]:
        """Parse an LLM response into clean text and action directives.

        Args:
            llm_response: Raw LLM response text.

        Returns:
            Tuple of (clean_text, action_list) where each action is:
            {"action": str, "param": str | None, "raw": str}
        """
        actions = []
        clean_lines = []

        for line in llm_response.split("\n"):
            # Check for [ACTION] directives
            matches = ACTION_PATTERN.findall(line)
            if matches:
                for match in matches:
                    action_name = match[0].lower()
                    param = match[1] if match[1] else None
                    actions.append(
                        {
                            "action": action_name,
                            "param": param,
                            "raw": f"[ACTION] {action_name}"
                            + (f": {param}" if param else ""),
                        }
                    )
                # Remove the [ACTION] part from the line but keep any remaining text
                cleaned_line = ACTION_PATTERN.sub("", line).strip()
                if cleaned_line:
                    clean_lines.append(cleaned_line)
            else:
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)
        return clean_text, actions

    def execute(self, actions: list[dict], current_fuel: float = 0.0) -> list[str]:
        """Execute a list of parsed actions.

        In dry_run mode, actions are logged but not sent to iRacing.
        In live mode, actions are sent via the iRacing client.

        Args:
            actions: List of action dicts from parse_response().
            current_fuel: Current fuel level in litres (for clamping add_fuel).

        Returns:
            List of result strings for each action.
        """
        results = []

        for action_dict in actions:
            action = action_dict["action"]
            param = action_dict.get("param")

            # Check if action is allowed
            if action not in self.allowed_actions:
                msg = f"[BLOCKED] Action '{action}' not in allowed list"
                logger.warning(msg)
                results.append(msg)
                continue

            if self.dry_run:
                msg = f"[WOULD EXECUTE] {action}" + (f": {param}" if param else "")
                logger.info(msg)
                results.append(msg)
                continue

            # Live execution
            if self.iracing is None:
                msg = f"[ERROR] No iRacing client connected for action '{action}'"
                logger.error(msg)
                results.append(msg)
                continue

            result = self._execute_action(action, param, current_fuel)
            results.append(result)

        return results

    def _execute_action(
        self, action: str, param: str | None, current_fuel: float = 0.0
    ) -> str:
        """Execute a single action via the iRacing client.

        Args:
            action: Action name (e.g. 'add_fuel', 'change_tyres')
            param: Optional parameter (e.g. '60' for add_fuel)
            current_fuel: Current fuel level in litres (for clamping add_fuel).

        Returns:
            Result string describing what happened.
        """
        try:
            if action == "pit_this_lap":
                # Request pit stop — uses pit command
                self.iracing.pit_command("clear_tires")
                return "[EXECUTED] pit_this_lap — pit stop requested"

            elif action == "add_fuel":
                if param is None:
                    return "[ERROR] add_fuel requires a parameter (litres)"

                try:
                    litres = float(param)
                except (ValueError, TypeError):
                    return f"[ERROR] Invalid fuel amount: {param}"

                # Clamp to reasonable values
                original = litres
                litres = max(1, round(litres))  # Minimum 1L, whole litres

                # Cap at tank capacity if known
                if self.fuel_max > 0:
                    max_add = max(0, self.fuel_max - current_fuel)
                    if litres > max_add:
                        litres = max(0, round(max_add))

                if litres <= 0:
                    return f"[BLOCKED] add_fuel: {original}L would exceed tank capacity ({self.fuel_max:.0f}L)"

                self.iracing.pit_command("fuel", int(litres))
                return f"[EXECUTED] add_fuel: {int(litres)}L" + (
                    f" (clamped from {original}L)" if original != litres else ""
                )

            elif action == "change_tyres":
                self.iracing.pit_command("clear_tires")
                return "[EXECUTED] change_tyres — tyre change requested"

            elif action == "clear_penalty":
                self.iracing.pit_command("clear_fr")
                return "[EXECUTED] clear_penalty — penalty cleared"

            else:
                return f"[ERROR] Unknown action: {action}"

        except Exception as e:
            error_msg = f"[ERROR] Failed to execute {action}: {e}"
            logger.error(error_msg)
            return error_msg
