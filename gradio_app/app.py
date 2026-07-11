"""A minimal Gradio app for the llm-engineering project.

``shout`` uppercases its input; ``build_demo`` wires it to an input textbox and
an output textbox. Keeping the demo behind a factory (rather than a module-level
value) means importing this package has no side effects, mirroring the lazy
client in ``openrouter/chat.py``.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from dotenv import load_dotenv


def shout(text: str) -> str:
    """Return ``text`` in uppercase."""
    return text.upper()


def build_demo() -> gr.Blocks:
    """Build the UI: a textbox in, its uppercase out."""
    return gr.Interface(
        fn=shout,
        inputs=gr.Textbox(label="Input"),
        outputs=gr.Textbox(label="Output"),
        title="llm-engineering",
        flagging_mode="never",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so API keys are available once the
    real UI is built, matching how ``main.py`` bootstraps the project.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
