"""Build a Chroma vector store over the local knowledge base.

This is the retriever that :mod:`rag.knowledge_base` deliberately is not. Where
that one keys each document by the words in its filename, this one chunks every
document, embeds the chunks, and retrieves by cosine similarity — so "what do
you do if a fret buzzes?" finds the setup document without naming it, and a
follow-up phrased around meaning rather than keywords still lands.

The pipeline is the usual four steps:

1. read every ``knowledge-base/<category>/<Name>.md`` into a LangChain
   ``Document``, tagged with the category as ``doc_type``;
2. split into overlapping character chunks, so a retrieved chunk is small
   enough to be mostly signal but large enough to stand on its own;
3. embed each chunk with a local sentence-transformers model — no API key, no
   per-token cost, and the vectors never leave the machine;
4. persist the lot to Chroma on disk, so the chat app starts instantly.

Loading goes through LangChain's ``DirectoryLoader`` and ``TextLoader`` rather
than reading the files directly. For flat Markdown that is more machinery than
``Path.read_text`` needs, but ``loader_cls`` is the extension point that makes
the rest of the pipeline source-agnostic: indexing PDFs alongside the Markdown
is ``loader_cls=PyPDFLoader`` and nothing else changes, because everything
downstream only sees ``Document`` objects.

Build or rebuild the store with ``uv run python -m rag.vector_store``; it is
cheap (seconds) and idempotent, dropping any existing collection first.
"""

from __future__ import annotations

from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.knowledge_base import KNOWLEDGE_BASE

# Where the embeddings live. Ignored by git: it is derived data, rebuildable
# from the Markdown in seconds.
VECTOR_DB = Path(__file__).parent / "vector_db"

# Small, fast, and local — 384 dimensions, ~90MB, downloaded once by
# sentence-transformers and cached in ~/.cache/huggingface.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Roughly a paragraph or two per chunk, with enough overlap that a fact split
# across a boundary still appears whole in one of the two chunks.
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


def load_documents(root: Path = KNOWLEDGE_BASE) -> list[Document]:
    """Read every ``*.md`` one level below ``root`` into a LangChain document.

    One ``DirectoryLoader`` per category folder, because the folder name is the
    only place the category is recorded — the loaders know nothing about that
    convention, so ``doc_type`` and ``name`` are attached afterwards. ``source``
    comes from ``TextLoader`` itself.

    ``name`` is the ``employees/Lena Okafor`` label a reply cites the chunk by;
    ``doc_type`` is ``company``, ``employees``, or ``products``.

    Both the folders and the files within them are sorted: ``DirectoryLoader``
    globs in filesystem order, which would make the contents of the store — and
    so the ids Chroma assigns — differ between machines for no reason.
    """
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    documents: list[Document] = []
    for folder in folders:
        loader = DirectoryLoader(
            str(folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for document in sorted(loader.load(), key=lambda d: d.metadata["source"]):
            document.metadata["doc_type"] = folder.name
            document.metadata["name"] = (
                f"{folder.name}/{Path(document.metadata['source']).stem}"
            )
            documents.append(document)
    if not documents:
        raise FileNotFoundError(f"no Markdown documents under {root}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving their metadata.

    ``RecursiveCharacterTextSplitter`` tries paragraph breaks before line
    breaks before spaces, so chunks tend to end at a blank line rather than
    mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def device() -> str:
    """Return a device this torch build can actually run kernels on.

    ``torch.cuda.is_available()`` is true for any CUDA GPU, including one whose
    compute capability the installed wheels were never compiled for — and that
    mismatch only surfaces as ``no kernel image is available`` on the first
    forward pass, halfway through indexing. Compare the card against the
    architectures the build ships instead, and fall back to CPU when it is
    older than all of them. The knowledge base is small enough that CPU
    embedding takes seconds either way.
    """
    if not torch.cuda.is_available():
        return "cpu"
    # "sm_86" is compute capability 8.6 and "sm_100" is 10.0 — the minor
    # version is the last digit, and everything before it is the major.
    architectures = [
        (int(digits[:-1]), int(digits[-1]))
        for arch in torch.cuda.get_arch_list()
        if (digits := arch.removeprefix("sm_")).isdigit()
    ]
    return "cuda" if torch.cuda.get_device_capability() >= min(architectures) else "cpu"


def embeddings(model: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Build the local embedding model used for both indexing and querying.

    Both sides must use the same model: a query vector is only comparable to
    chunk vectors from the same embedding space.

    ``.env`` is loaded here so ``HF_TOKEN`` is in the environment before
    sentence-transformers resolves the model on the Hub. The model is public and
    works without it — the token only buys higher rate limits and faster
    downloads, and silences the Hub's unauthenticated-request warning.
    """
    load_dotenv()
    return HuggingFaceEmbeddings(model=model, model_kwargs={"device": device()})


def build(root: Path = KNOWLEDGE_BASE, persist_directory: Path = VECTOR_DB) -> Chroma:
    """Index ``root`` into a fresh Chroma collection at ``persist_directory``.

    Any existing collection is dropped first, so a rebuild after editing the
    knowledge base never leaves stale chunks behind alongside the new ones.
    """
    chunks = split_documents(load_documents(root))
    embedding = embeddings()
    if persist_directory.exists():
        Chroma(
            persist_directory=str(persist_directory), embedding_function=embedding
        ).delete_collection()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(persist_directory),
    )


def open_store(persist_directory: Path = VECTOR_DB) -> Chroma:
    """Open the persisted store, refusing to open one that was never built.

    Chroma will happily hand back an empty collection for a directory that does
    not exist, which shows up much later as a chatbot that has never heard of
    Fretwork. Failing here says what to run instead.
    """
    store = Chroma(
        persist_directory=str(persist_directory), embedding_function=embeddings()
    )
    if not store.get(include=[])["ids"]:
        raise FileNotFoundError(
            f"no vectors in {persist_directory} — build it with "
            f"`uv run python -m rag.vector_store`"
        )
    return store


def main() -> None:
    """Build the store and report what went into it."""
    documents = load_documents()
    chunks = split_documents(documents)
    doc_types = sorted({document.metadata["doc_type"] for document in documents})
    print(f"Loaded {len(documents)} documents ({', '.join(doc_types)})")
    print(f"Split into {len(chunks)} chunks of up to {CHUNK_SIZE} characters")

    vectors = build().get(include=["embeddings"])["embeddings"]
    print(
        f"Indexed {len(vectors):,} vectors of {len(vectors[0]):,} "
        f"dimensions into {VECTOR_DB}"
    )


if __name__ == "__main__":
    main()
