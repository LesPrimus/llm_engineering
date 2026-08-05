"""Simple retrieval-augmented generation over a local Markdown knowledge base.

Two retrievers over the same documents: keyword matching on filenames
(:mod:`rag.knowledge_base`, served by :mod:`rag.chat`) and embedding similarity
over chunks (:mod:`rag.vector_store`, served by :mod:`rag.vector_chat`).

Only the keyword pair is re-exported here. The vector modules pull in torch,
sentence-transformers, and chromadb at import time, which is seconds of startup
that ``import rag`` should not cost anyone who only wants the small version —
import ``rag.vector_chat`` directly instead.
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
