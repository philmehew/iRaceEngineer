"""
OpenAI-compatible LLM client — sends condensed race context to any
OpenAI-compatible endpoint and returns the response.

Works with OpenAI, Ollama Cloud, Ollama Local, LM Studio, or any
OpenAI-compatible API by changing base_url in config.
"""

import logging
import os
import re
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails (timeout, network, API error).

    Callers should catch this and show/speak a short user-facing message,
    rather than treating the error text as an engineer response.
    """


def _is_transient(e: Exception) -> bool:
    """Whether an API error is worth retrying (rate limits, server overload,
    connection issues) vs a permanent failure (auth, bad request, not found).
    """
    # openai SDK exposes typed errors; fall back to string matching for
    # anything raised before the SDK classes it (e.g. httpx connection errors)
    try:
        import openai

        if isinstance(
            e,
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(e, (openai.AuthenticationError, openai.BadRequestError)):
            return False
    except ImportError:
        pass
    err = str(e).lower()
    return any(
        s in err
        for s in ("rate limit", "429", "503", "overloaded", "connection", "network")
    )


class LLMClient:
    """Send race context to an LLM and return the response.

    Configuration is read from config.yaml under the 'llm' key:
        base_url:       API endpoint (default: Ollama Cloud)
        api_key_env:    Environment variable name for the API key
        model:          Model name to use
        max_tokens:     Max response tokens
        temperature:    Response randomness (0 = deterministic)
        retries:        Retry attempts for transient errors (default 2)
        retry_backoff:  Base backoff between retries in seconds (default 1.0)
        thinking:      Thinking control — false to disable, "low"/"medium"/"high"
                       to set depth, empty/unset to use model default
    """

    def __init__(self, config: dict):
        llm_config = config.get("llm", {})

        # Resolve API key — supports both patterns:
        #   api_key_env: "OLLAMA_API_KEY"   → reads from environment variable
        #   api_key_env: "sk-abc123..."     → uses the value directly as the key
        #   api_key: "sk-abc123..."         → explicit key field (takes precedence)
        api_key = llm_config.get("api_key", "")
        if not api_key:
            api_key_env_val = llm_config.get("api_key_env", "OLLAMA_API_KEY")
            # If the value matches a valid env-var name (identifier-ish with
            # underscores), treat it as an env var name; otherwise use it
            # directly as the key. Env var names are UPPER_SNAKE identifiers —
            # real keys contain dots, hyphens, or lowercase/mixed case.
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", api_key_env_val):
                api_key = os.environ.get(api_key_env_val, "")
            else:
                api_key = api_key_env_val

        base_url = llm_config.get("base_url", "https://api.ollama.com/v1")
        model = llm_config.get("model", "ministral-3:14b-cloud")
        max_tokens = llm_config.get("max_tokens", 300)
        temperature = llm_config.get("temperature", 0.3)

        self.client = OpenAI(
            api_key=api_key or "placeholder",  # Some endpoints don't require a key
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = llm_config.get("timeout", 5.0)
        self.retries = llm_config.get("retries", 2)
        self.retry_backoff = llm_config.get("retry_backoff", 1.0)
        # Optional thinking/reasoning depth control. Sent via extra_body with
        # both known spellings — "think" (Ollama, honours false|"low"|"medium"|
        # "high") and "reasoning_effort" (OpenAI, "low"|"medium"|"high") — each
        # endpoint uses whichever it recognises and ignores the other.
        #   thinking: false     → disable thinking (Ollama) / "low" (OpenAI)
        #   thinking: "low"     → light reasoning
        #   thinking: "high"    → deep reasoning
        #   thinking: "" / unset → don't send anything (model default)
        raw_thinking = llm_config.get("thinking", "")
        if isinstance(raw_thinking, bool):
            self.thinking: bool | str | None = raw_thinking
        elif isinstance(raw_thinking, str) and raw_thinking.strip().lower() in (
            "false",
            "off",
            "none",
        ):
            self.thinking = False
        elif isinstance(raw_thinking, str) and raw_thinking.strip():
            self.thinking = raw_thinking.strip().lower()
        else:
            self.thinking = None
        self.base_url = base_url
        self._api_key = api_key

        # Log key status without revealing the actual key
        key_status = "set directly" if api_key else "missing"
        logger.info(
            f"LLM client initialised: {base_url} model={model} api_key={key_status}"
        )

    def ask(self, messages: list[dict], question: str = "") -> str:
        """Send messages to the LLM and return the response text.

        Args:
            messages: List of message dicts for the chat completions API.
            question: Optional follow-up question to append.

        Returns:
            The assistant's response text.

        Raises:
            LLMError: If all attempts fail (timeout, network, API error).
        """
        # If a direct question is provided, append it
        if question:
            messages = messages + [{"role": "user", "content": question}]

        last_error: Exception | None = None
        for attempt in range(1 + self.retries):
            if attempt > 0:
                delay = self.retry_backoff * attempt
                logger.info(f"LLM retry {attempt}/{self.retries} in {delay:.1f}s")
                time.sleep(delay)

            try:
                t_start = time.monotonic()
                create_kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                }
                if self.thinking is not None:
                    # Endpoints have no standard spelling for this — pass both
                    # known keys; the endpoint uses whichever it recognises.
                    # Ollama's "think" accepts false|true|"low"|"medium"|"high";
                    # OpenAI's "reasoning_effort" only accepts strings, so a
                    # disabled-thinking request maps to "low" there.
                    create_kwargs["extra_body"] = {
                        "think": self.thinking,
                        "reasoning_effort": (
                            "low" if self.thinking is False else self.thinking
                        ),
                    }
                response = self.client.chat.completions.create(**create_kwargs)
                t_done = time.monotonic()

                message = response.choices[0].message
                content = (message.content or "").strip()

                # Handle "thinking" models (e.g. DeepSeek R1, ornith1) that put
                # reasoning in a separate field — fall back to reasoning_content
                # if the main content is empty.
                if not content:
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        content = reasoning.strip()
                        logger.info(
                            "LLM response was in reasoning_content (thinking model)"
                        )

                # Log token usage and timing
                if hasattr(response, "usage") and response.usage:
                    logger.info(
                        f"LLM response: {response.usage.total_tokens} tokens "
                        f"({response.usage.prompt_tokens} prompt, "
                        f"{response.usage.completion_tokens} completion) "
                        f"in {t_done - t_start:.3f}s"
                    )
                else:
                    logger.info(f"LLM response in {t_done - t_start:.3f}s")

                return content

            except Exception as e:
                last_error = e
                err_str = str(e)
                timed_out = (
                    "timeout" in err_str.lower() or "timed out" in err_str.lower()
                )
                if timed_out:
                    logger.warning(f"LLM call timed out (attempt {attempt + 1})")
                else:
                    logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                # Only retry transient errors — auth/invalid-request failures
                # won't fix themselves, so fail fast instead of stalling.
                if not timed_out and not _is_transient(e):
                    break

        # If the call timed out, give a user-friendly message
        err_str = str(last_error) if last_error else ""
        if "timeout" in err_str.lower() or "timed out" in err_str.lower():
            logger.warning("LLM call timed out")
            raise LLMError("I'm busy, try again in a minute.") from last_error
        logger.error(f"LLM call failed: {err_str}")
        raise LLMError(f"LLM call failed: {err_str}") from last_error

    def ask_streaming(self, messages: list[dict], question: str = ""):
        """Send messages to the LLM and yield response chunks.

        Useful for real-time display of the response as it arrives.

        Args:
            messages: List of message dicts for the chat completions API.
            question: Optional follow-up question to append.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        if question:
            messages = messages + [{"role": "user", "content": question}]

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Handle thinking models that send reasoning_content separately
                text = delta.content or getattr(delta, "reasoning_content", None) or ""
                if text:
                    yield text

        except Exception as e:
            logger.error(f"LLM streaming call failed: {e}")
            yield f"[Error] {e}"
