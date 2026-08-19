"""Answer questions from the LLM-preprocessed Chroma store, via OpenRouter.

The retrieval half of the pipeline whose indexing half is :mod:`rag.ingest`.
That module asks an LLM to split each document into chunks and to write a
headline and a summary for every one, then embeds the lot into
``preprocessed_db``. This one queries that store and turns what comes back into
an answer.

Where :mod:`rag.vector_chat` embeds the question, takes the top five chunks,
and generates, this one spends four extra calls to earn a better context:

1. **rewrite** — a follow-up that dropped its subject ("does it come in blue?")
   embeds to nothing useful, because the words that would find the right
   document are two turns back. An LLM folds the history into a single
   self-contained query.
2. **retrieve twice** — once for what the user actually typed, once for the
   rewrite, ``RETRIEVAL_K`` chunks each. The rewrite is a guess and it can
   drift; keeping the original question's hits alongside it costs one embedding
   call and means a bad rewrite degrades the context instead of replacing it.
3. **merge** — union of the two, original question's hits first, deduplicated
   on chunk text.
4. **rerank** — the ~40 survivors go to an LLM as a numbered list, which
   returns them in relevance order. This is the step embeddings cannot do:
   a similarity score compares two vectors that were computed without ever
   seeing each other, while the reranker reads the question and the chunk
   together.
5. **answer** — the top ``FINAL_K`` become the context for the reply.

Everything is self-contained: this module shares no code with :mod:`rag.ingest`,
only three constants' worth of agreement about where the store is and how it
was embedded. Those three are marked below, and they are the ones to change in
both places or not at all — a query vector is only comparable to chunk vectors
from the same embedding model.

Index first, then ask::

    uv run python -m rag.ingest
    uv run python -m rag.answer "which acoustic is best for small hands?"
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The three constants shared with rag.ingest by agreement rather than by import:
# where the store lives, what the collection is called, and which model wrote
# the vectors in it. Changing one here without changing it there gets you an
# empty collection or an incomparable query vector.
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "openai/text-embedding-3-large"

# LiteLLM needs the "openrouter/" prefix to pick the provider — a bare
# "openai/..." would go straight to OpenAI. The embedding client below is
# pointed at OpenRouter by base_url instead, so it takes the slug unprefixed,
# as OpenRouter itself spells it.
#
# Three models, because the three jobs are graded on different things. The
# answering model writes the prose the user reads, so it is the good one. The
# other two are mechanical and sit in the latency path before the first token,
# so they stay small — but not equally small.
#
# Reranking is pure instruction-following: read a numbered list, return the
# numbers in a different order. Nano does it, and it is the model rag.ingest
# already relies on for structured output.
#
# Rewriting needs to track a referent across turns, and nano is below the floor
# for that. Given "Which acoustic is best for small hands?" / "the Skylark
# Parlour" / "and in black?", nano drops the subject and rewrites to "which
# acoustic guitar is available in black at Fretwork"; mini keeps it. That is a
# failed retrieval on every elliptical follow-up, which is most of them in a
# real conversation, so it is worth the tier.
MODEL = "openrouter/anthropic/claude-sonnet-4.5"
REWRITE_MODEL = "openrouter/openai/gpt-4.1-mini"
RERANK_MODEL = "openrouter/openai/gpt-4.1-nano"

# How many chunks each of the two searches returns, and how many survive the
# rerank into the prompt. Retrieval is deliberately loose and the rerank is
# what tightens it: recall is cheap here (one vector search) and precision is
# expensive (a whole generation spent on the wrong passage), so over-fetch and
# let the reranker throw work away.
RETRIEVAL_K = 20
FINAL_K = 10

# Someone is waiting on this, unlike the ingestion pipeline's patient
# minutes-long backoff. Retry a rate limit or a dropped connection a few times,
# quickly, then give up and say so rather than hang.
retrying = retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)

SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant for Fretwork, an online guitar store, answering questions for its staff and customers.

Answer only from the extracts below. They are the whole of what you know about Fretwork, and they change from question to question. They are excerpts rather than whole documents, so one may cut off mid-thought — use what is there and do not fill in the rest. If they do not contain the answer, say so plainly and say what you would need; never guess at a price, a spec, a date, or a person. Quote figures exactly as they appear.

Your answer will be judged on accuracy, relevance and completeness, so answer the question that was asked, and answer all of it.

Here are the extracts from the knowledge base most likely to be relevant:

{context}
"""

REWRITE_PROMPT = """You are about to search a knowledge base about Fretwork, an online guitar store, to answer a user's question.

This is the conversation so far:
{history}

And this is the user's current question:
{question}

The conversation is contextual, so the current question may only make sense given what came before. Work out what the user actually means and write it out in full, adding the details the question leaves implicit. Condense it into a single very short, very specific search query, most likely to surface the passage that answers it.

EXAMPLE:
user: Who founded the company? -> Query: who founded Fretwork?
assistant: Fretwork was founded by FooBar.
user: What does she play? -> Query: which guitar does FooBar, founder of Fretwork, play?

If the conversation is empty, or it does not tell you what the question refers to, write the question out as a search query as it stands. Never invent a subject it did not name, and never write a placeholder such as "[product name]" or "[the item the user means]" — the query is searched literally, so a placeholder finds nothing.

IMPORTANT: Respond ONLY with the search query, nothing else.
"""

RERANK_PROMPT = """You are a document re-ranker.
You are provided with a question and a list of chunks of text retrieved from a knowledge base.
The chunks are given in the order they were retrieved, which is approximately ordered by relevance, but you can probably improve on that.
Rank the chunks by how well they answer the question, most relevant first.
Include every chunk id you are given, exactly once. Reply only with the list of ranked chunk ids, nothing else.
"""


class Result(BaseModel):
    """A retrieved chunk: the text that was embedded, and where it came from.

    The same shape :mod:`rag.ingest` writes — ``page_content`` is the headline,
    summary, and original text the chunking model produced, and ``metadata``
    carries the ``source`` path and ``type`` folder.
    """

    page_content: str
    metadata: dict[str, Any]


class RankOrder(BaseModel):
    """The reranker's reply: chunk ids, best first."""

    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to "
        "least relevant, by chunk id number"
    )


@cache
def open_client() -> OpenAI:
    """Build the embedding client, pointed at OpenRouter.

    OpenRouter serves embeddings on the same OpenAI-compatible wire protocol,
    so a base URL plus a key is the whole integration. The SDK would otherwise
    reach for ``OPENAI_API_KEY``, which this project never sets, so the key is
    resolved explicitly.

    Built on first use rather than at import, so merely importing this module
    cannot fail on a missing key.
    """
    return OpenAI(
        base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]
    )


@cache
def open_collection() -> Collection:
    """Open the persisted collection, refusing to open one that was never built.

    Chroma hands back a new empty collection for a store that does not exist,
    which surfaces much later as an assistant that has never heard of Fretwork.
    Failing here says what to run instead.
    """
    collection = PersistentClient(path=DB_NAME).get_or_create_collection(
        collection_name
    )
    if not collection.count():
        raise FileNotFoundError(
            f"no vectors in {DB_NAME} — build it with `uv run python -m rag.ingest`"
        )
    return collection


def _turns(history: Sequence[Mapping[str, Any]]) -> Iterator[tuple[str, str]]:
    """Yield the ``(role, content)`` pairs worth replaying from a history.

    Gradio history entries are ``{"role": ..., "content": ...}`` dicts, but the
    content of a turn can also be a file or a tuple, and either side can be
    absent. Only text turns from the user and the assistant survive.
    """
    for turn in history:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            yield role, content


def _label(metadata: Mapping[str, Any]) -> str:
    """Render a chunk's provenance as ``products/Skylark``.

    ``rag.ingest`` records ``source`` as an absolute path, which is noise in a
    prompt and worse in a citation line — the folder and the filename are the
    parts that identify the document.
    """
    stem = Path(str(metadata.get("source", "unknown"))).stem
    doc_type = str(metadata.get("type", ""))
    return f"{doc_type}/{stem}" if doc_type else stem


@retrying
def _complete(
    model: str,
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None = None,
) -> str:
    """Call a model through LiteLLM and return its reply text.

    The single seam every LLM call in this module goes through, so the retry
    policy is declared once. An empty reply is raised rather than returned:
    it fails the same way a dropped connection does, and it should be retried
    the same way too.
    """
    response = completion(
        model=model, messages=messages, response_format=response_format
    )
    reply = response.choices[0].message.content  # type: ignore[union-attr]
    if not reply:
        raise ValueError(f"{model} returned an empty response")
    return reply


def rewrite_query(question: str, history: Sequence[Mapping[str, Any]] = ()) -> str:
    """Fold the conversation into one self-contained search query.

    The history is rendered as a transcript rather than interpolated as a list
    of dicts: the model reads ``user: ...`` / ``assistant: ...`` better than it
    reads Python ``repr``, and the braces and quotes are tokens spent on
    nothing.
    """
    transcript = (
        "\n".join(f"{role}: {content}" for role, content in _turns(history))
        or "(nothing yet — this is the first question)"
    )
    prompt = REWRITE_PROMPT.format(history=transcript, question=question)
    return _complete(REWRITE_MODEL, [{"role": "user", "content": prompt}]).strip()


def fetch_context_unranked(question: str) -> list[Result]:
    """Embed ``question`` and return the ``RETRIEVAL_K`` nearest chunks.

    The bare vector goes to ``query``, not a list holding it: Chroma accepts one
    embedding or many and casts the single case up, and the results come back
    nested per query either way — hence the ``[0]`` on each.
    """
    vector = (
        open_client()
        .embeddings.create(model=embedding_model, input=[question])
        .data[0]
        .embedding
    )
    results = open_collection().query(query_embeddings=vector, n_results=RETRIEVAL_K)
    documents = (results["documents"] or [[]])[0]
    metadatas = (results["metadatas"] or [[]])[0]
    return [
        Result(page_content=document, metadata=dict(metadata or {}))
        for document, metadata in zip(documents, metadatas, strict=True)
    ]


def merge_chunks(
    primary: Sequence[Result], secondary: Sequence[Result]
) -> list[Result]:
    """Union of two retrievals, ``primary`` first, deduplicated on chunk text.

    Chunks overlap by design — :mod:`rag.ingest` asks for about 25% overlap
    between neighbours — but identical text means the same chunk, and the two
    searches usually agree on several of them. ``seen`` grows as it goes, so a
    chunk repeated within ``secondary`` is added once rather than twice.

    Order matters downstream: the reranker is told its input is roughly ordered
    already, so the hits for what the user literally asked go in front of the
    hits for the rewrite.
    """
    merged = list(primary)
    seen = {chunk.page_content for chunk in merged}
    for chunk in secondary:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            merged.append(chunk)
    return merged


def rerank(question: str, chunks: Sequence[Result]) -> list[Result]:
    """Reorder ``chunks`` by relevance to ``question``, most relevant first.

    Chunks are numbered from 1 in the prompt and the model replies with the ids
    alone, which keeps the reply short and makes it impossible for the reranker
    to edit the text it is ranking.

    Everything after the call is defence against that reply being wrong, which
    a structured-output schema constrains the *shape* of but not the contents:
    ids outside the range are dropped, repeats are ignored, and chunks the
    model forgot are appended in their retrieval order. A reply that will not
    parse at all falls back to the retrieval order — the ranking is a
    refinement, and losing it is not worth losing the answer over.
    """
    if not chunks:
        return []
    user_prompt = (
        f"The user has asked the following question:\n\n{question}\n\n"
        "Order all the chunks of text by relevance to the question, from most "
        "relevant to least relevant. Include all the chunk ids you are provided "
        "with, reranked.\n\nHere are the chunks:\n\n"
    )
    for index, chunk in enumerate(chunks, start=1):
        user_prompt += f"# CHUNK ID: {index}\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": RERANK_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        reply = _complete(RERANK_MODEL, messages, response_format=RankOrder)
        order = RankOrder.model_validate_json(reply).order
    except ValidationError, ValueError:
        return list(chunks)

    ranked: list[Result] = []
    seen: set[int] = set()
    for chunk_id in order:
        if 1 <= chunk_id <= len(chunks) and chunk_id not in seen:
            seen.add(chunk_id)
            ranked.append(chunks[chunk_id - 1])
    ranked.extend(
        chunk for index, chunk in enumerate(chunks, start=1) if index not in seen
    )
    return ranked


def fetch_context(
    question: str, history: Sequence[Mapping[str, Any]] = ()
) -> list[Result]:
    """Run the whole retrieval pipeline and return the ``FINAL_K`` best chunks.

    The rerank is judged against the rewrite, not against the question as
    typed. Ranking reads the question and the chunk together, so a follow-up
    that dropped its subject gives the reranker nothing to read: ask "does it
    come in a left-handed version?" and every chunk looks equally plausible,
    so it demotes the very chunks the rewrite just went and found. The rewrite
    is that same question with its subject put back, which is what relevance
    has to be measured against.

    The searches stay split, though, and that is where a bad rewrite is caught:
    the question as typed contributes its own ``RETRIEVAL_K`` candidates, so a
    rewrite that drifted can misorder the shortlist but cannot empty it.
    """
    rewritten = rewrite_query(question, history)
    chunks = merge_chunks(
        fetch_context_unranked(question), fetch_context_unranked(rewritten)
    )
    return rerank(rewritten, chunks)[:FINAL_K]


def make_rag_messages(
    question: str,
    history: Sequence[Mapping[str, Any]],
    chunks: Sequence[Result],
) -> list[dict[str, str]]:
    """Assemble the request: grounded system prompt, prior turns, new question.

    Each extract is labelled with the document it came from, which both helps
    the model attribute a fact and lets it say "the Skylark page doesn't cover
    that" instead of a bare refusal.
    """
    context = (
        "\n\n".join(
            f"# Extract from {_label(chunk.metadata)}\n\n{chunk.page_content.strip()}"
            for chunk in chunks
        )
        or "(nothing was retrieved for this question)"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *({"role": role, "content": content} for role, content in _turns(history)),
        {"role": "user", "content": question},
    ]


def answer_question(
    question: str, history: Sequence[Mapping[str, Any]] = ()
) -> tuple[str, list[Result]]:
    """Answer ``question`` from the knowledge base.

    Returns the answer and the chunks it was grounded in, so a caller can cite
    them, show them, or score them.
    """
    chunks = fetch_context(question, history)
    return _complete(MODEL, make_rag_messages(question, history, chunks)), chunks


def main() -> None:
    """Answer a question from the command line, citing what it retrieved."""
    question = " ".join(sys.argv[1:]) or "Which acoustic is best for small hands?"
    print(f"Q: {question}\n")
    answer, chunks = answer_question(question)

    ###############################################################################

    sources = dict.fromkeys(_label(chunk.metadata) for chunk in chunks)
    print(answer)
    print(f"\n---\nContext: {', '.join(sources) or 'nothing retrieved'}")


if __name__ == "__main__":
    main()
