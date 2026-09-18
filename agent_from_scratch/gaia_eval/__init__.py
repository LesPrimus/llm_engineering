"""Evaluating the agent on the GAIA benchmark — questions, answers, scoring."""

from .constants import Level, Split
from .dataset import GaiaDataset
from .models import AnnotatorMetadata, GaiaReply, GaiaTask
from .results import Attempt, Scorecard, by_level, summarise
from .scoring import is_correct

__all__ = [
    "AnnotatorMetadata",
    "Attempt",
    "GaiaDataset",
    "GaiaReply",
    "GaiaTask",
    "Level",
    "Scorecard",
    "Split",
    "by_level",
    "is_correct",
    "summarise",
]
