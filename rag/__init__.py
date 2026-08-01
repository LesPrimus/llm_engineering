"""Simple retrieval-augmented generation over a local Markdown knowledge base."""

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
