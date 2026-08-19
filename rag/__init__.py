"""Simple retrieval-augmented generation over a local Markdown knowledge base.

Three retrievers over the same documents, in order of how much work they do
before answering:

- keyword matching on filenames (:mod:`rag.knowledge_base`, served by
  :mod:`rag.chat`);
- embedding similarity over fixed-size chunks, split and embedded locally
  (:mod:`rag.vector_store`, served by :mod:`rag.vector_chat`);
- embedding similarity over LLM-written chunks, each carrying a generated
  headline and summary, then reranked by an LLM before answering
  (:mod:`rag.ingest`, served by :mod:`rag.answer`).

Only the keyword pair is re-exported here. The other modules pull in torch,
sentence-transformers, chromadb, and litellm at import time, which is seconds of
startup that ``import rag`` should not cost anyone who only wants the small
version — import ``rag.vector_chat`` or ``rag.answer`` directly instead.
"""

from .chat import Bot, build_demo, launch
from .knowledge_base import KNOWLEDGE_BASE, Document, KnowledgeBase

__all__ = [
    "KNOWLEDGE_BASE",
    "Bot",
    "Document",
    "KnowledgeBase",
    "build_demo",
    "launch",
]
