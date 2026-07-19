"""A Gradio text-to-speech app: type text and the pipeline speaks it.

Runs a text-to-speech pipeline locally on the GPU. Type a sentence, click
Speak, and the generated audio plays back in the browser. The default model
(facebook/mms-tts-eng) is a small public VITS model, so no HF token is needed;
its weights are downloaded to ~/.cache/huggingface on first run.

    uv run python -m hugging_face.tts_app
"""

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

MODEL = "facebook/mms-tts-eng"

# Built once at import: downloads/caches the weights and loads them onto the GPU,
# so each speak() call is just a fast forward pass.
tts = pipeline("text-to-speech", model=MODEL, device="cuda")


def speak(text: str) -> tuple[int, np.ndarray] | None:
    """Synthesize ``text`` and return (sample_rate, waveform) for gr.Audio."""
    text = text.strip()
    if not text:
        return None
    out = tts(text)
    # The pipeline returns float32 in [-1, 1] shaped (1, N); squeeze to mono and
    # convert to int16 PCM, which gr.Audio plays without a conversion warning.
    audio = np.squeeze(out["audio"])
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return out["sampling_rate"], pcm


demo = gr.Interface(
    fn=speak,
    inputs=gr.Textbox(label="Text", placeholder="Type something to speak..."),
    outputs=gr.Audio(label="Speech", autoplay=True),
    title="llm-engineering — local text-to-speech",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
