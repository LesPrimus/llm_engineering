"""A toy sentiment-analysis pipeline running locally on the GPU.

Wraps a text-classification pipeline in a self-contained dataclass that owns its
pipeline. The default model (distilbert-base-uncased-finetuned-sst-2-english) is a
public SST-2 sentiment model, so no HF token is needed; its weights are downloaded
to ~/.cache/huggingface on first run.

    uv run python -m hugging_face.sentiment
"""

from dataclasses import dataclass, field

from dotenv import load_dotenv
from transformers import Pipeline, pipeline

load_dotenv()

# Canonical lightweight sentiment model (POSITIVE/NEGATIVE); also the pipeline's
# implicit default for text classification, so setting it explicitly only silences
# the "no model supplied" warning.
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass
class SentimentAnalyzer:
    """Classifies the sentiment of a piece of text, owning the pipeline it uses."""

    model: str = DEFAULT_MODEL
    device: str = "cuda"
    # "text-classification" is the canonical task; "sentiment-analysis" is an alias
    # for it, but only the canonical names are in pipeline()'s typed overloads.
    _pipeline: Pipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pipeline = pipeline(
            "text-classification", model=self.model, device=self.device
        )

    def __call__(self, text: str) -> list[dict[str, object]]:
        return self._pipeline(text)


def main() -> None:
    analyzer = SentimentAnalyzer()
    print(analyzer("I'm super excited!"))


if __name__ == "__main__":
    main()
