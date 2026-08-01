"""Retrieval-augmented chat over a small local knowledge base, via OpenRouter.

The smallest thing that is still RAG: for each question, look up the documents
whose names appear in it (:mod:`rag.knowledge_base`), paste them into the system
prompt, and let the model answer from that context alone. No embeddings, no
vector store — the point is the *augmentation* step, not the retriever.

The knowledge base describes Fretwork, a fictional online guitar store: its
company overview, four staff, and four product lines. Ask "what does the
Meridian cost?" or "who is Okafor?" and the matching document is in context;
ask something the knowledge base does not name and the model is told so and
should say it doesn't know.

Every reply ends with the documents that were retrieved for it, so it is
obvious when an answer is grounded and when the model was working blind.

The model is reached through an OpenAI client pointed at OpenRouter, like
``gradio_app.multibot`` — the client factory calls ``load_dotenv`` before
reading ``OPENROUTER_API_KEY``, and the bot is built inside ``build_demo`` (not
at import), so importing this module has no side effects.

Run it with ``uv run python -m rag.chat`` from the project root, or straight
from an IDE run button — the knowledge base is imported absolutely
(``rag.knowledge_base``), so this file also works when executed as a script,
provided the project root is on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)

from rag.knowledge_base import Document, KnowledgeBase

# Marks the citation line appended to each reply. It is stripped back off when a
# reply is replayed as history, so the model never sees its own footers.
SOURCES_PREFIX = "\n\n---\n_Context: "


def _client() -> OpenAI:
    # Load .env here so OPENROUTER_API_KEY is present. The OpenAI SDK's default
    # env var is OPENAI_API_KEY, so point the client at OpenRouter and resolve
    # the key explicitly rather than relying on the SDK fallback.
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@dataclass(frozen=True)
class Bot:
    """A chatbot that answers from retrieved documents and nothing else."""

    SYSTEM: ClassVar[str] = (
        "You answer questions about Fretwork, an online guitar store, for its "
        "staff and customers. Answer only from the context below: it is the "
        "whole of what you know about Fretwork, and it changes from question to "
        "question. If the context does not contain the answer, say so plainly "
        "and say what you would need — never guess at a price, a spec, a date, "
        "or a person. Keep answers short and concrete, and quote figures exactly "
        "as they appear."
    )

    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase.load)
    model: str = "anthropic/claude-sonnet-4.5"
    client: OpenAI = field(default_factory=_client)
    system: str = SYSTEM

    def _system_message(self, documents: Sequence[Document]) -> str:
        """Build the system prompt: the persona plus the retrieved documents.

        With nothing retrieved the model is told the context is empty, which is
        what keeps it from answering a Fretwork question from its own priors.
        """
        if not documents:
            return f"{self.system}\n\nRelevant context: none found for this question."
        blocks = "\n\n".join(
            f"# {document.name}\n\n{document.text.strip()}" for document in documents
        )
        return f"{self.system}\n\nRelevant context:\n\n{blocks}"

    def _messages(
        self,
        message: str,
        history: Sequence[dict[str, Any]],
        documents: Sequence[Document],
    ) -> list[Message]:
        """Map the retrieved context, prior turns, and new question to a request.

        Gradio history entries are ``{"role": ..., "content": ...}`` dicts;
        only ``user``/``assistant`` turns with text are kept, and each assistant
        turn is trimmed back to the reply itself, without its sources footer.
        """
        messages: list[Message] = [
            SystemMessage(role="system", content=self._system_message(documents))
        ]
        for turn in history:
            content = turn.get("content")
            if not isinstance(content, str) or not content:
                continue
            if turn.get("role") == "user":
                messages.append(UserMessage(role="user", content=content))
            elif turn.get("role") == "assistant":
                messages.append(
                    AssistantMessage(
                        role="assistant", content=content.split(SOURCES_PREFIX)[0]
                    )
                )
        messages.append(UserMessage(role="user", content=message))
        return messages

    def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
        """Retrieve context for ``message``, then stream the grounded answer.

        Yields the reply accumulated so far on each chunk, so
        ``gr.ChatInterface`` renders it as it arrives, then one final time with
        the retrieved documents cited underneath.

        Retrieval runs against the new message only — not the history — so a
        follow-up has to name what it asks about to pull that document back in.
        """
        documents = self.knowledge.search(message)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(message, history, documents),
            stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                yield reply
        yield reply + _sources(documents)


def _sources(documents: Sequence[Document]) -> str:
    """Render the citation line listing what retrieval found."""
    names = (
        ", ".join(document.name for document in documents)
        if documents
        else "nothing retrieved"
    )
    return f"{SOURCES_PREFIX}{names}_"


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn chat grounded in the local knowledge base."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — RAG over the Fretwork knowledge base",
        description=(
            "Ask about Fretwork, its staff (Okafor, Reyes, Raghunathan, Delgado), "
            "or its products (Meridian, Driftwood, Lowtide, Roundhouse). Naming one "
            "of them puts that document in the model's context; each reply cites "
            "what was retrieved."
        ),
        examples=[
            "What does the Meridian cost, and what are the finishes?",
            "What does Reyes own at Fretwork?",
            "What is Fretwork's returns policy?",
            "Is the Driftwood or the Meridian better for fingerstyle?",
        ],
    )


def launch(**kwargs: Any) -> None:
    """Build the demo and serve it.

    The bot's client factory calls ``load_dotenv`` as it builds, so
    ``OPENROUTER_API_KEY`` is read from ``.env`` without ``launch`` having to
    manage the environment.
    """
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
