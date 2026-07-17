"""A minimal Gradio text-to-speech: type a message and hear it played back.

No bot / LLM reply — whatever you type is spoken aloud via OpenRouter TTS
(``deepgram/aura-2``, fast). Typed messages show in a transcript on the left; the reply
audio plays in the Voice player on the right. One OpenAI-SDK client pointed at OpenRouter
(``OPENROUTER_API_KEY``).

A plain ``gr.Blocks`` with a single (non-streaming) submit event feeds
``gr.Audio(autoplay=True)`` — ``gr.ChatInterface``'s streaming ``additional_outputs``
triggered Gradio autoplay bugs.
Run it with ``uv run python -m gradio_app.chat_voice``.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

TTS_MODEL = "deepgram/aura-2"
TTS_VOICE = "aura-2-thalia-en"


def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@dataclass
class Bot:
    """Speaks whatever text it is given — no chat / LLM reply."""

    client: OpenAI = field(default_factory=_client)
    tts_model: str = TTS_MODEL
    tts_voice: str = TTS_VOICE

    def talker(self, message):
        response = self.client.audio.speech.create(
            model=self.tts_model, voice=self.tts_voice, input=message
        )
        return response.content


def build_demo(bot: Bot | None = None) -> gr.Blocks:
    active = bot if bot is not None else Bot()

    def on_submit(
        message: str, history: list[dict[str, str]]
    ) -> (
        tuple[list[dict[str, str]], None, str] | tuple[list[dict[str, str]], bytes, str]
    ):
        """Append the typed message to the transcript, speak it, and clear the box."""
        message = message.strip()
        if not message:
            return history, None, ""
        history = [*history, {"role": "user", "content": message}]
        return history, active.talker(message), ""

    with gr.Blocks(title="llm-engineering — type & hear") as demo:
        gr.Markdown("# Type a message — hear it spoken")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat")
                textbox = gr.Textbox(
                    placeholder="Type something and press Enter", show_label=False
                )
            with gr.Column(scale=1):
                audio = gr.Audio(label="Voice", autoplay=True)
        textbox.submit(on_submit, [textbox, chatbot], [chatbot, audio, textbox])
    return demo


def launch(**kwargs: Any) -> None:
    load_dotenv()
    kwargs.setdefault("allowed_paths", [tempfile.gettempdir()])
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
