"""
OpenAI-compatible LLM client — sends condensed race context to any
OpenAI-compatible endpoint and returns the response.

Works with OpenAI, Ollama Cloud, Ollama Local, LM Studio, or any
OpenAI-compatible API by changing base_url in config.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Send race context to an LLM and return the response.

    Configuration is read from config.yaml under the 'llm' key:
        base_url:     API endpoint (default: Ollama Cloud)
        api_key_env:  Environment variable name for the API key
        model:        Model name to use
        max_tokens:   Max response tokens
        temperature:  Response randomness (0 = deterministic)
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
            # If the value looks like an API key (contains a dot or is long),
            # use it directly rather than treating it as an env var name
            if "." in api_key_env_val or len(api_key_env_val) > 40:
                api_key = api_key_env_val
            else:
                api_key = os.environ.get(api_key_env_val, "")

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
            The assistant's response text, or an error message if the call fails.
        """
        # If a direct question is provided, append it
        if question:
            messages = messages + [{"role": "user", "content": question}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            content = response.choices[0].message.content.strip()

            # Log token usage
            if hasattr(response, "usage") and response.usage:
                logger.info(
                    f"LLM response: {response.usage.total_tokens} tokens "
                    f"({response.usage.prompt_tokens} prompt, "
                    f"{response.usage.completion_tokens} completion)"
                )

            return content

        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            logger.error(error_msg)
            return f"[Error] {error_msg}"

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
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming call failed: {e}")
            yield f"[Error] {e}"
