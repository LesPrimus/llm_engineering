"""A minimal Gradio text-to-speech: type a message and hear it played back.

No bot / LLM reply — whatever you type is spoken aloud via OpenRouter TTS
(``deepgram/aura-2``, fast). Typed messages show in a transcript on the left; the reply
audio plays in the Voice player on the right. One OpenAI-SDK client pointed at OpenRouter
(``OPENROUTER_API_KEY``).

Two things make ``autoplay`` play from the very start instead of clipping the first word:
(1) a plain ``gr.Blocks`` with a single (non-streaming) submit event feeding
``gr.Audio(autoplay=True)`` — ``gr.ChatInterface``'s streaming ``additional_outputs``
triggered Gradio autoplay bugs; and (2) **uncompressed** audio — TTS is requested as
``pcm`` (Deepgram returns a WAV), decoded to samples, and handed to ``gr.Audio`` as
``(sample_rate, samples)``, so there is no MP3 decode latency for autoplay to skip past.
Run it with ``uv run python -m gradio_app.chat_voice``.
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from gradio import processing_utils
from openai import OpenAI

TTS_MODEL = "deepgram/aura-2"
TTS_VOICE = "aura-2-thalia-en"

# Laptop speaker amplifiers power down when idle and take ~0.5-1s to wake, clipping the
# start of playback (headphones have no such amp and are unaffected). Prepend this much
# silence so the amp wakes during the silence, not during the first word. Tunable.
LEAD_IN_SECONDS = 1.0


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

    def speak(self, text: str) -> tuple[int, Any] | None:
        """Synthesize ``text`` and return ``(sample_rate, samples)`` (``None`` on failure,
        so a TTS error clears the player instead of breaking the UI).

        Requests uncompressed audio (``pcm`` → Deepgram returns a WAV) and decodes it to
        samples. Handing ``gr.Audio`` raw samples (served as WAV) means playback has **no
        decode latency**, so ``autoplay`` starts at sample 0 instead of clipping the front
        of the speech the way a compressed ``mp3`` does.
        """
        try:
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text,
                response_format="pcm",
            )
            fd, path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as handle:
                handle.write(response.content)
            sample_rate, samples = processing_utils.audio_from_file(path)
            pad = np.zeros(
                (int(sample_rate * LEAD_IN_SECONDS), *samples.shape[1:]),
                dtype=samples.dtype,
            )
            return sample_rate, np.concatenate([pad, samples], axis=0)
        except Exception as exc:
            print(f"[chat_voice] TTS failed: {exc}", file=sys.stderr)
            return None


def build_demo(bot: Bot | None = None) -> gr.Blocks:
    active = bot if bot is not None else Bot()

    def on_submit(
        message: str, history: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], tuple[int, Any] | None, str]:
        """Append the typed message to the transcript, speak it, and clear the box."""
        message = message.strip()
        if not message:
            return history, None, ""
        history = [*history, {"role": "user", "content": message}]
        return history, active.speak(message), ""

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
