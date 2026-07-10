"""An adversarial two-model conversation run entirely through OpenRouter.

GPT plays a snarky, argumentative bot while Claude plays a polite, agreeable
one. Both models are reached as OpenRouter model IDs, so this module never
touches the Anthropic SDK or the client classes in ``client.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)

load_dotenv()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Built lazily (not at import) so load_dotenv() has run before we read the key.
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def _reply_text(response: ChatCompletion) -> str:
    """Pull the assistant's text out of an OpenRouter response."""
    return response.choices[0].message.content or ""


@dataclass(frozen=True)
class Bot:
    """One participant: how it's labelled, which model runs it, and its persona."""

    label: str
    model: str
    system: str

    def chat(self, conversation: list[tuple[Bot, str]]) -> str:
        """Ask this bot for its next line, given the conversation so far. Each
        past turn becomes an ``assistant`` message if this bot said it, or a
        ``user`` message if the other bot did."""
        messages: list[Message] = [SystemMessage(role="system", content=self.system)]
        for speaker, text in conversation:
            if speaker is self:
                messages.append(AssistantMessage(role="assistant", content=text))
            else:
                messages.append(UserMessage(role="user", content=text))
        response = _client().chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return _reply_text(response)


GPT = Bot(
    label="GPT",
    model="openai/gpt-4o-mini",
    system=(
        "You are a chatbot who is very argumentative; you disagree with anything "
        "in the conversation and you challenge everything, in a snarky way."
    ),
)
CLAUDE = Bot(
    label="Claude",
    model="anthropic/claude-sonnet-4.5",
    system=(
        "You are a very polite, courteous chatbot. You try to agree with everything "
        "the other person says, or find common ground. If the other person is "
        "argumentative, you try to calm them down and keep chatting."
    ),
)


def run_conversation(rounds: int = 5) -> None:
    """Run ``rounds`` back-and-forths between GPT and Claude, printing each turn."""
    conversation: list[tuple[Bot, str]] = [(GPT, "Hi there"), (CLAUDE, "Hi")]
    for bot, text in conversation:
        print(f"{bot.label}:\n{text}\n")

    for _ in range(rounds):
        for bot in (GPT, CLAUDE):
            reply = bot.chat(conversation)
            conversation.append((bot, reply))
            print(f"{bot.label}:\n{reply}\n")


if __name__ == "__main__":
    run_conversation()
