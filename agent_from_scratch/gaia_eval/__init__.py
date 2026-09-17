"""Evaluating the agent on the GAIA benchmark — questions, answers, scoring."""

from .dataset import GaiaDataset, Split
from .models import AnnotatorMetadata, GaiaReply, GaiaTask

__all__ = ["AnnotatorMetadata", "GaiaDataset", "GaiaReply", "GaiaTask", "Split"]
