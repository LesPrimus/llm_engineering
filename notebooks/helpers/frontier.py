"""Price a product with a frontier model that has never seen this dataset.

A `FrontierPricer` is a callable of one `Item` returning the model's raw reply,
which is exactly the shape `notebooks.helpers.evaluator.evaluate` expects -- its
`post_process` pulls the first number out of a string, so "around $1,249.99"
scores the same as a float. Nothing here computes an error or draws a chart; the
evaluator owns all of that.

    from notebooks.helpers.frontier import FrontierPricer, Model

    pricer = FrontierPricer(Model.Claude)
    evaluate(pricer, test, size=100, title=Model.Claude.name)

Every model is reached through one OpenAI client pointed at OpenRouter, so
adding a vendor is one more line in `Model`.

Two things this module exists to get right:

- **Reasoning is capped, not the reply.** All three models think before they
  answer, and left alone they will spend hundreds of output tokens deciding what
  a $20 cable costs. `REASONING` holds that down -- and it is where the money
  goes: Claude answers in about 6 output tokens, Gemini in a couple of hundred,
  because Gemini will not let reasoning be switched off at all. The tempting
  fix -- a small `max_tokens` -- is the wrong one: reasoning is drawn from the
  same budget, so the cap truncates the answer instead, `content` comes back
  empty, and the evaluator scores a silent guess of $0. A working model then
  reads as a broken one on the chart.
- **A blank reply is recorded, not swallowed.** Requests are retried on the
  transient API failures, since one 429 inside the evaluator's thread pool would
  otherwise kill a whole run. A reply that still carries no digit afterwards is
  appended to `blanks` so the notebook can say how many of the scored zeros were
  the model being wrong versus the model saying nothing.
"""

import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from notebooks.models.items import Item

BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = (
    "You estimate what a product costs. "
    "Reply with the price in US dollars and nothing else."
)
USER = "How much does this cost?\n\n{description}\n\nPrice:"

# The cheapest setting all three accept, sent through OpenRouter's normalized
# `reasoning` field for each vendor to map to its own knob. Turning reasoning
# off outright ({"enabled": False}) is cheaper still on Claude and GPT, but
# Gemini rejects it with "Reasoning is mandatory for this endpoint", and one
# setting for everyone keeps the three scores comparable.
REASONING = {"effort": "minimal"}

TRANSIENT = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class Model(Enum):
    """The models under test. Member name is the label; value is the OpenRouter ID."""

    Claude = "anthropic/claude-sonnet-5"
    GPT = "openai/gpt-5.2"
    Gemini = "google/gemini-3.1-pro-preview"


def _client() -> OpenAI:
    # Load .env here so OPENROUTER_API_KEY is present. The OpenAI SDK's default
    # env var is OPENAI_API_KEY, so point the client at OpenRouter and resolve
    # the key explicitly rather than relying on the SDK fallback.
    load_dotenv()
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


@dataclass(frozen=True)
class FrontierPricer:
    """One model, asked the price of an item with no training and no examples."""

    model: Model
    client: OpenAI = field(default_factory=_client)
    system: str = SYSTEM
    blanks: list[str] = field(default_factory=list)

    def messages(self, item: Item) -> list[Message]:
        """Build the two messages sent for `item`.

        The description is `item.full`: `summary` exists only on the training
        split, so using it would price the test items from text they do not
        have.
        """
        return [
            SystemMessage(role="system", content=self.system),
            UserMessage(role="user", content=USER.format(description=item.full or "")),
        ]

    @retry(
        retry=retry_if_exception_type(TRANSIENT),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def ask(self, item: Item) -> str:
        """Send one request and return the reply text, retrying transient failures."""
        response = self.client.chat.completions.create(
            model=self.model.value,
            messages=self.messages(item),
            extra_body={"reasoning": REASONING},
        )
        return response.choices[0].message.content or ""

    def __call__(self, item: Item) -> str:
        """Return the raw reply, recording it if the model answered with no number."""
        reply = self.ask(item)
        if not any(character.isdigit() for character in reply):
            self.blanks.append(item.title)
        return reply
