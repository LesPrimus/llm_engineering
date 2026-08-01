"""A tiny local knowledge base with keyword retrieval.

Documents are Markdown files under ``rag/knowledge-base/<category>/<Name>.md``.
Each one is keyed by the words in its own filename, so ``employees/Lena
Okafor.md`` is retrieved by "lena" or "okafor" and ``products/Meridian.md`` by
"meridian". Retrieval is then a set intersection between those keys and the
words of the question — no embeddings, no vector store, no chunking.

That is deliberately the crudest thing that still counts as retrieval: it makes
the *augmentation* step in RAG visible without any of the machinery. It also
fails in the obvious ways — a question that names nothing in the knowledge base
retrieves nothing, and a pronoun ("what does she work on?") retrieves nothing
either, because the key has to appear in the message itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_BASE = Path(__file__).parent / "knowledge-base"

# Short words make noisy keys ("a", "of"), so filename words below this length
# are not indexed. Every document still needs at least one longer word.
MIN_KEY_LENGTH = 3


def _words(text: str) -> list[str]:
    """Split ``text`` into lowercase alphabetic words, dropping punctuation.

    ``"Who is Okafor, and what's the Meridian?"`` becomes ``["who", "is",
    "okafor", "and", "what", "s", "the", "meridian"]`` — so "Meridian?" and
    "Okafor," still match their document keys.
    """
    letters = "".join(ch if ch.isalpha() else " " for ch in text)
    return letters.lower().split()


@dataclass(frozen=True)
class Document:
    """One Markdown file, plus the words that retrieve it."""

    path: Path
    text: str
    keys: frozenset[str]

    @classmethod
    def load(cls, path: Path) -> Document:
        """Read ``path`` and key it by the words of its filename."""
        keys = {word for word in _words(path.stem) if len(word) >= MIN_KEY_LENGTH}
        if not keys:
            raise ValueError(f"{path} has no filename word long enough to index")
        return cls(
            path=path,
            text=path.read_text(encoding="utf-8"),
            keys=frozenset(keys),
        )

    @property
    def name(self) -> str:
        """Human label such as ``employees/Lena Okafor``, used to cite the doc."""
        return f"{self.path.parent.name}/{self.path.stem}"


@dataclass(frozen=True)
class KnowledgeBase:
    """Every document under a knowledge-base root, searchable by keyword."""

    documents: tuple[Document, ...]

    @classmethod
    def load(cls, root: Path = KNOWLEDGE_BASE) -> KnowledgeBase:
        """Load every ``*.md`` file one level below ``root``.

        Paths are sorted so retrieval results come back in a stable order
        (category, then filename) rather than in filesystem order.
        """
        paths = sorted(root.glob("*/*.md"))
        if not paths:
            raise FileNotFoundError(f"no Markdown documents under {root}")
        return cls(tuple(Document.load(path) for path in paths))

    def search(self, query: str) -> list[Document]:
        """Return every document whose keys appear as words in ``query``."""
        words = set(_words(query))
        return [document for document in self.documents if document.keys & words]
