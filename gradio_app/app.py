"""A placeholder Gradio app for the llm-engineering project.

This is a scaffold: ``build_demo`` returns an empty-ish ``gr.Blocks`` UI meant
to be replaced with a real interface later. Keeping the demo behind a factory
(rather than a module-level value) means importing this package has no side
effects, mirroring the lazy client in ``openrouter/chat.py``.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from dotenv import load_dotenv


def build_demo() -> gr.Blocks:
    """Build and return the Gradio UI. Replace the placeholder with real UI."""
    with gr.Blocks(title="llm-engineering") as demo:
        gr.Markdown("# llm-engineering\nPlaceholder — replace me with a real UI.")
    return demo


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so API keys are available once the
    real UI is built, matching how ``main.py`` bootstraps the project.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
