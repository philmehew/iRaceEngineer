"""
Chat macro manager — maps LLM-declared actions to iRacing chat macros.

The iRacing SDK broadcast API can fire chat macros (AutoChatStr1-15 in
app.ini) but cannot send arbitrary chat text. So LLM "actions" are resolved
to macro numbers here, and the macro itself carries the chat command
(e.g. "!clearall$") — the trailing "$" in app.ini makes iRacing auto-transmit
without needing an Enter keypress.

Protocol: the LLM includes [CMD:action_name] tags in its reply when the
driver asks for an action. This module strips the tags (so they're never
spoken by TTS) and fires the corresponding macro via the SDK broadcast.
"""

import configparser
import logging
import os
import re

logger = logging.getLogger(__name__)

# [CMD:action_name] tags embedded in LLM responses
CMD_TAG_RE = re.compile(r"\[CMD:\s*(\w+)\s*\]")


class ChatMacroManager:
    """Resolve config-declared actions to iRacing chat macro numbers.

    Config (config.yaml):

        chat_macros:
          actions:
            clear_black_flag:
              macro: 15                 # AutoChatStr slot in iRacing (1-15)
              command: "!clearall"      # expected macro content for validation
              description: "..."        # shown to the LLM so it knows when to use it
    """

    def __init__(self, config: dict | None = None):
        self.actions: dict[str, dict] = {}
        cm_config = (config or {}).get("chat_macros", {})
        actions = cm_config.get("actions", {})
        for name, spec in actions.items():
            macro = spec.get("macro")
            if not isinstance(macro, int) or not 1 <= macro <= 15:
                logger.warning(
                    f"chat macro action '{name}': macro must be an integer 1-15, "
                    f"got {macro!r} — action disabled"
                )
                continue
            self.actions[name] = {
                "macro": macro,
                "command": spec.get("command", ""),
                "description": spec.get("description", ""),
            }
        if self.actions:
            logger.info(f"Chat macro actions: {', '.join(self.actions)}")
        else:
            logger.debug("No chat macro actions configured")

    @property
    def enabled(self) -> bool:
        """True when at least one action is configured."""
        return bool(self.actions)

    def _app_ini_path(self, config: dict | None) -> str:
        """Locate app.ini — explicit config path or the default Documents location."""
        configured = (config or {}).get("chat_macros", {}).get("app_ini_path")
        if configured:
            return os.path.expanduser(configured)
        return os.path.join(
            os.environ.get("USERPROFILE", os.path.expanduser("~")),
            "Documents",
            "iRacing",
            "app.ini",
        )

    def validate_against_app_ini(self, config: dict | None = None) -> list[str]:
        """Check that each action's macro slot contains the expected text.

        app.ini is only a snapshot — the sim loads it at startup and writes it
        back on exit, so a mid-session UI edit won't appear here. Mismatches
        are warnings, not failures: the macro still fires; only its content
        differs from what config promises.

        Returns:
            List of warning strings (empty when all macros validate).
        """
        warnings: list[str] = []
        path = self._app_ini_path(config)
        if not os.path.isfile(path):
            warnings.append(f"app.ini not found at {path} — cannot validate macros")
            logger.warning(warnings[0])
            return warnings

        parser = configparser.ConfigParser(
            inline_comment_prefixes=(";",), strict=False, interpolation=None
        )
        try:
            parser.read(path, encoding="utf-8")
        except Exception as e:
            warnings.append(f"Failed to read app.ini ({e}) — cannot validate macros")
            logger.warning(warnings[0])
            return warnings

        section = "Autochat Messages"
        for name, spec in self.actions.items():
            key = f"AutoChatStr{spec['macro']}"
            if not parser.has_option(section, key):
                warnings.append(f"{name}: {key} missing from app.ini")
                continue
            # Strip the trailing "$" (auto-transmit marker) and inline comments
            actual = parser.get(section, key).strip()
            if actual.endswith("$"):
                actual = actual[:-1]
            expected = spec["command"]
            if expected.endswith("$"):
                expected = expected[:-1]
            if expected and actual != expected:
                # app.ini lags the sim's in-memory macros until the sim exits
                # and writes the file back — a mismatch may just mean the slot
                # was edited in the UI since the last sim restart
                warnings.append(
                    f"{name}: macro {spec['macro']} contains '{actual}' "
                    f"but config expects '{expected}' "
                    f"(app.ini may be stale if the sim edited it recently)"
                )

        for w in warnings:
            logger.warning(f"Chat macro validation: {w}")
        if not warnings:
            logger.info(
                f"All {len(self.actions)} chat macro action(s) validated against app.ini"
            )
        return warnings

    def available_actions_prompt(self) -> str:
        """Render the actions block for the system prompt (empty if none)."""
        if not self.actions:
            return ""
        lines = [
            "You can execute commands for the driver. Available commands:",
        ]
        for name, spec in self.actions.items():
            desc = f" - {spec['description']}" if spec["description"] else ""
            lines.append(f"{name}{desc}")
        lines.append(
            "Only when the driver explicitly asks you to execute a command, "
            "include its tag in your reply (e.g. [CMD:"
            + next(iter(self.actions))
            + "]) "
            "along with your spoken answer. Never invent command names."
        )
        return "\n".join(lines)

    def parse_response(self, text: str) -> tuple[str, list[str]]:
        """Extract [CMD:name] tags from an LLM response.

        Returns:
            (clean_text, action_names) — clean text has tags stripped so they
            are never displayed or spoken; unknown action names are dropped
            with a warning.
        """
        tags = CMD_TAG_RE.findall(text or "")
        clean = CMD_TAG_RE.sub("", text or "").strip()
        valid: list[str] = []
        for name in tags:
            if name in self.actions:
                valid.append(name)
            else:
                logger.warning(f"LLM referenced unknown command '{name}' — ignored")
        return clean, valid

    def fire(self, action_name: str, iracing_client=None) -> bool:
        """Fire the macro for an action.

        Args:
            action_name: Configured action name.
            iracing_client: IRacingClient to send through. None (replay/tests)
                logs the intent instead of sending — the pipeline can be
                exercised end-to-end without iRacing.

        Returns:
            True when the macro was sent (or would have been, in log-only mode).
        """
        spec = self.actions.get(action_name)
        if spec is None:
            logger.warning(f"Cannot fire unknown chat macro action: {action_name}")
            return False
        if iracing_client is None:
            logger.info(
                f"[replay/no-sim] Would fire chat macro {spec['macro']} "
                f"for action '{action_name}'"
            )
            return True
        try:
            iracing_client.send_chat_macro(spec["macro"])
            logger.info(f"Fired chat macro {spec['macro']} for action '{action_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to fire chat macro for '{action_name}': {e}")
            return False
