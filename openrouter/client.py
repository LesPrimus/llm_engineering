import os

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam


class OpenRouterClient:
    """Call any model through OpenRouter's OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None) -> None:
        # OpenAI's default env var is OPENAI_API_KEY, so resolve OpenRouter's
        # key explicitly rather than relying on the SDK's fallback.
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    def ask(self, model: str, prompt: str, provider: str | None = None) -> str:
        extra_body = {}
        if provider is not None:
            # Pin the request to one OpenRouter provider instead of letting it
            # pick the host by price/uptime; fail rather than fall back elsewhere.
            extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
        response = self._client.chat.completions.create(
            model=model,
            messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""
