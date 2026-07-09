from anthropic import Anthropic
from anthropic.types import MessageParam


class ClaudeClient:
    """Call Claude models directly through the Anthropic API."""

    def __init__(self, api_key: str | None = None) -> None:
        # Anthropic() reads ANTHROPIC_API_KEY from the environment when
        # api_key is None, so pass it through either way.
        self._client = Anthropic(api_key=api_key)

    def ask(self, model: str, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=prompt)],
        )
        return next(
            (block.text for block in response.content if block.type == "text"), ""
        )
