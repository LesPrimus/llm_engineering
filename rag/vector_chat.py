"""Retrieval-augmented chat over the Chroma vector store, via OpenRouter.

The same idea as :mod:`rag.chat` with a real retriever underneath. That module
retrieves whole documents by filename keyword; this one retrieves the ``K``
nearest *chunks* by embedding similarity, so a question can find the right
passage without naming the document it lives in — "how long does a setup take?"
lands on the setup standard, and "who should I ask about pricing?" lands on
Ruth Feldman.

The two pieces LangChain contributes are a ``retriever`` and an ``llm``, both of
which answer to ``invoke``:

    retriever.invoke(question) -> list[Document]   # nearest chunks
    llm.invoke(messages)       -> AIMessage        # the answer

Everything between them is a format string. That is the whole of RAG.

The LLM is OpenRouter, reached through ``ChatOpenAI`` — OpenRouter speaks the
OpenAI wire protocol, so pointing ``base_url`` at it and passing
``OPENROUTER_API_KEY`` is the entire integration, and any model on OpenRouter is
one string away. Embeddings stay local (:mod:`rag.vector_store`), so only the
question and the retrieved chunks ever leave the machine.

Build the store first, then run the chat::

    uv run python -m rag.vector_store
    uv run python -m rag.vector_chat
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag.vector_store import open_store

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4.5"

# How many chunks to put in front of the model. Low enough that the context
# stays mostly relevant, high enough that a question spanning two documents
# ("is the Skylark or the Driftwood better for small hands?") sees both.
K = 5

# Marks the citation line appended to each reply. It is stripped back off when a
# reply is replayed as history, so the model never sees its own footers.
SOURCES_PREFIX = "\n\n---\n_Context: "


def _retriever(k: int = K) -> VectorStoreRetriever:
    """Open the persisted store and wrap it as a top-``k`` retriever."""
    return open_store().as_retriever(search_kwargs={"k": k})


def _llm(model: str = MODEL) -> ChatOpenAI:
    # Load .env here so OPENROUTER_API_KEY is present. ChatOpenAI would
    # otherwise reach for OPENAI_API_KEY, so both the key and the base URL are
    # passed explicitly rather than left to the default resolution.
    load_dotenv()
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        # A grounded answer should read the context back, not improvise around
        # it, so sample as close to deterministically as the model allows.
        temperature=0,
    )


@dataclass(frozen=True)
class Bot:
    """A chatbot that answers from retrieved chunks and nothing else."""

    SYSTEM: ClassVar[str] = (
        "You answer questions about Fretwork, an online guitar store, for its "
        "staff and customers. Answer only from the context below: it is the "
        "whole of what you know about Fretwork, and it changes from question to "
        "question. The context is excerpts, not whole documents, so it may cut "
        "off mid-thought — use what is there and do not fill in the rest. If the "
        "context does not contain the answer, say so plainly and say what you "
        "would need — never guess at a price, a spec, a date, or a person. Keep "
        "answers short and concrete, and quote figures exactly as they appear."
        "\n\nContext:\n\n{context}"
    )

    retriever: VectorStoreRetriever = field(default_factory=_retriever)
    llm: ChatOpenAI = field(default_factory=_llm)
    system: str = SYSTEM

    def _system_message(self, documents: Sequence[Document]) -> str:
        """Fill the persona template with the retrieved chunks.

        Each chunk is labelled with the document it came from, which both helps
        the model attribute a fact and lets it say "the Meridian page doesn't
        cover that" instead of a bare refusal.
        """
        if not documents:
            return self.system.format(context="none found for this question.")
        blocks = "\n\n".join(
            f"# {document.metadata.get('name', 'unknown')}\n\n{document.page_content.strip()}"
            for document in documents
        )
        return self.system.format(context=blocks)

    def _messages(
        self,
        message: str,
        history: Sequence[dict[str, Any]],
        documents: Sequence[Document],
    ) -> list[BaseMessage]:
        """Map the retrieved context, prior turns, and new question to a request.

        Gradio history entries are ``{"role": ..., "content": ...}`` dicts; only
        ``user``/``assistant`` turns with text are kept, and each assistant turn
        is trimmed back to the reply itself, without its sources footer.
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=self._system_message(documents))
        ]
        for turn in history:
            content = turn.get("content")
            if not isinstance(content, str) or not content:
                continue
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=content))
            elif turn.get("role") == "assistant":
                messages.append(AIMessage(content=content.split(SOURCES_PREFIX)[0]))
        messages.append(HumanMessage(content=message))
        return messages

    def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
        """Retrieve chunks for ``message``, then stream the grounded answer.

        Yields the reply accumulated so far on each chunk, so
        ``gr.ChatInterface`` renders it as it arrives, then one final time with
        the retrieved documents cited underneath.

        Retrieval runs against the new message only — not the history — so a
        follow-up phrased as "and in blue?" searches for those words alone.
        Similarity search is more forgiving of that than keyword matching, but
        it is not a rewriter: it still finds whatever the words are near.
        """
        documents = self.retriever.invoke(message)
        reply = ""
        for chunk in self.llm.stream(self._messages(message, history, documents)):
            reply += chunk.text
            yield reply
        yield reply + _sources(documents)


def _sources(documents: Sequence[Document]) -> str:
    """Render the citation line: which documents the chunks came from.

    Several chunks often come from one document, so names are de-duplicated
    while keeping them in rank order — nearest match first.
    """
    names = dict.fromkeys(
        document.metadata.get("name", "unknown") for document in documents
    )
    listed = ", ".join(names) if names else "nothing retrieved"
    return f"{SOURCES_PREFIX}{listed}_"


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn chat grounded in the vector store."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — vector RAG over the Fretwork knowledge base",
        description=(
            "Ask about Fretwork — its company policies, its staff, or its "
            "instruments. Each question retrieves the nearest passages from the "
            f"Chroma store (top {K}), so you do not have to name the document "
            "you want; every reply cites what was retrieved."
        ),
        examples=[
            "What happens if I don't get on with the guitar I ordered?",
            "How long does a setup take, and what does it involve?",
            "Who should I ask about pricing?",
            "Which acoustic is best for small hands?",
        ],
    )


def launch(**kwargs: Any) -> None:
    """Build the demo and serve it.

    Both the retriever and the LLM are built as the bot is constructed — the
    first opens the persisted store (and raises if it was never built), the
    second calls ``load_dotenv`` and reads ``OPENROUTER_API_KEY``.
    """
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
