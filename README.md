# llm-engineering

Experiments with LLMs through [OpenRouter](https://openrouter.ai), using the OpenAI SDK pointed at OpenRouter's OpenAI-compatible API. One key, many models: `main.py` sends the same prompt to several models (GPT, Claude, Gemini, Llama, DeepSeek) and prints their answers side by side.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root (it's gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
uv run main.py
```

To try other models, add their IDs from <https://openrouter.ai/models> to the `MODELS` list in `main.py`.

## Web UI

Two Gradio apps live in the `gradio_app` package; both open a local page in your
browser.

**Single-model (Claude).** Type a message and Claude Sonnet's reply streams back
through the Anthropic API:

```bash
uv run python -m gradio_app.app
```

It reads `ANTHROPIC_API_KEY` from your `.env`, so add that key alongside
`OPENROUTER_API_KEY`.

**Multibot selector.** Pick GPT, Claude, or DeepSeek from a dropdown; the reply
streams back through OpenRouter:

```bash
uv run python -m gradio_app.multibot
```

It uses `OPENROUTER_API_KEY` — no Anthropic key needed.

**Website summarizer.** Paste a company's website URL and pick GPT, Claude, or
DeepSeek; the landing page is fetched with BeautifulSoup and a short Markdown
summary streams back:

```bash
uv run python -m gradio_app.website_summarizer
```

It uses `OPENROUTER_API_KEY` too. Only the given page is fetched — links are not
followed.

**Airline ticket chat.** A multi-turn airline-ticket assistant built on
`gr.ChatInterface`. Ask what a flight costs and Claude calls a `get_airline_price`
tool that looks the price up in a small SQLite database and returns it in euros;
the reply streams back through the Anthropic API:

```bash
uv run python -m gradio_app.airline
```

It reads `ANTHROPIC_API_KEY` from your `.env` (native Claude, like the
single-model app above) — no OpenRouter key needed. Prices are placeholder data
in a local SQLite database (`airline_prices.db`, created and seeded on launch by
an injected `PriceStore`), not a real fare lookup.

**Airline assistant with image & voice.** A multimodal take on the airline
assistant with three boxes — chat, a generated image of the destination city, and
a spoken (TTS) rendering of every reply. Everything runs through a single
OpenAI-SDK client pointed at OpenRouter: chat + function calling
(`openai/gpt-4o-mini`), the price tool (the same SQLite dummy prices), image
generation (`openai/gpt-image-1`), and voice (`hexgrad/kokoro-82m` TTS —
OpenRouter has no OpenAI TTS model). Ask to fly somewhere and the bot calls
`get_airline_price`, generates an image of that city, and speaks the reply:

```bash
uv run python -m gradio_app.airline_multimodal
```

It reads `OPENROUTER_API_KEY` from your `.env` — one key covers all three
modalities. Self-contained (no imports from `gradio_app.airline`); the image
(PIL) and voice (bytes) outputs are served straight from Gradio's cache, so there
are no temp files.

## RAG

The `rag` package holds two retrievers over the same knowledge base — a keyword
one and a vector one — behind two otherwise identical chat apps, so the
difference the retriever makes is the only thing that changes.

### Keyword retrieval

`rag.chat` is the smallest thing that is still retrieval-augmented generation:
for each question it looks up the documents whose names appear in it, pastes
them into the system prompt, and lets the model answer from that context alone.
No embeddings, no vector store, no chunking — the point is the augmentation
step, not the retriever.

```bash
uv run python -m rag.chat
```

Run it from the project root. `rag/chat.py` imports the knowledge base
absolutely (`rag.knowledge_base`), so the file also runs as a plain script from
an IDE run button — PyCharm puts the content root on `PYTHONPATH` by default. A
bare `python rag/chat.py` from the shell does not, and needs
`PYTHONPATH=. python rag/chat.py` or the `-m` form above.

It reads `OPENROUTER_API_KEY` from your `.env` and routes the chat through
OpenRouter (`anthropic/claude-sonnet-4.5` by default — change `Bot.model` for
another model ID).

The knowledge base is `rag/knowledge-base/`, Markdown files under `company/`,
`employees/`, and `products/` describing Fretwork, a fictional online guitar
store. Each document is keyed by the words in its own filename, so
`employees/Lena Okafor.md` is retrieved by "lena" or "okafor" and
`products/Meridian.md` by "meridian"; retrieval is a set intersection between
those keys and the words of the question. Drop another Markdown file into one of
those folders and it is retrievable by its filename immediately.

Every reply cites the documents that were retrieved for it, so it is visible
when an answer is grounded and when the model was working blind — and the
retriever's limits are visible too. A question that names nothing in the
knowledge base retrieves nothing, and so does a follow-up that leans on a
pronoun ("what does she work on?"), because retrieval sees only the new message.
With an empty context the model is told so, and should say it doesn't know
rather than invent a price.

### Vector retrieval

`rag.vector_chat` is the same app with an actual retriever underneath, built on
LangChain: every document is split into overlapping ~1000-character chunks,
embedded locally with sentence-transformers `all-MiniLM-L6-v2`, and stored in
Chroma on disk. A question retrieves the five nearest chunks by cosine
similarity, so it no longer has to name the document it wants — "what happens if
I don't get on with the guitar I ordered?" finds the returns policy, and "who
should I ask about pricing?" finds Ruth Feldman.

Build the store once, then chat:

```bash
uv run python -m rag.vector_store   # ~40 chunks, seconds
uv run python -m rag.vector_chat
```

Rebuild after editing the knowledge base — the build drops the old collection
first, so it never leaves stale chunks behind. The store lands in
`rag/vector_db/` and is git-ignored: it is derived data.

The LLM is OpenRouter again, this time through LangChain's `ChatOpenAI` pointed
at `https://openrouter.ai/api/v1` — OpenRouter speaks the OpenAI wire protocol,
so the base URL plus `OPENROUTER_API_KEY` is the whole integration, and any
model on OpenRouter is one string away (`MODEL` in `rag/vector_chat.py`).
Embeddings stay local, so only the question and the retrieved chunks leave the
machine.

Embedding runs on the GPU when torch has kernels for it and on the CPU when it
does not — `vector_store.device()` compares the card's compute capability
against `torch.cuda.get_arch_list()`, because `torch.cuda.is_available()` also
returns true for a GPU too old for the installed wheels, and that only shows up
as `no kernel image is available` partway through indexing.

Replies cite documents rather than chunks, de-duplicated in rank order, so the
citation line stays readable when several chunks come from one file. Each
question is searched for twice — once on its own, and once against the whole
conversation joined together — and the two result sets are merged in rank order,
so a follow-up like "and in blue?" can still find the document whose name is two
turns back without the new question losing its own best matches.

### Evaluating it

Two questions, measured separately: did the right documents come back, and did
the model then use them?

`rag.evaluation` answers the first, against 32 hand-written cases in
`rag/eval_cases.jsonl`. Nothing is generated and the same question returns the
same chunks every time, so it costs nothing to run and a sweep over settings
costs seconds. Five numbers come out — hit rate, recall, precision, MRR and nDCG
— reported per kind of case, because `single`, `followup`, `cross` and `refusal`
are not equally hard and averaging them together hides the failures.

`rag.judge` answers the second, and this one spends money: each case is answered
by the bot for real and graded by a *different* model (`google/gemini-2.5-pro`
grading `anthropic/claude-sonnet-4.5`, because a model marking its own homework
grades generously). It returns four booleans per case. `correct` and `grounded`
are deliberately separate: a reply that is right but ungrounded came from the
model's pretraining rather than from the retrieved context, which means
retrieval is not doing the work and the case will turn into a confident wrong
answer the day the knowledge base disagrees with the model.

```bash
uv run python -m rag.evaluation   # free
uv run python -m rag.judge        # ~71 model calls over the full set
```

`rag.eval_app` puts both behind a Gradio dashboard, which is where the knobs
become legible:

```bash
uv run python -m rag.eval_app
```

The **Retrieval** tab has `k` and `k_conversation` as sliders and a sweep that
scores every `k` from 1 to 12 and plots the five metrics against it — recall
climbing to a plateau while precision falls away, which is the trade the setting
actually is. The whole sweep is free. The **Answers** tab picks which kinds to
grade before spending anything, says how many model calls the run will cost, and
streams rows in as each case is judged rather than making you wait for the run
to end. The **Cases** tab is the labelled set itself.

Both tabs need `OPENROUTER_API_KEY`, and the store built, before they will start.

## Notebooks

**Open-model tour** (`notebooks/hf_open_models_tour.ipynb`). Runs five open Hugging
Face models — Llama 3.2 1B, Phi-4-mini, Gemma 3 270M, Qwen3-4B and the
DeepSeek-R1-Distill reasoning model — against the same prompt, at the tokenizer /
`AutoModelForCausalLM` level rather than behind a `pipeline`. Each is loaded in
4-bit and freed again before the next, so all five fit comfortably.

Built for **Google Colab on a T4 GPU**, not for local execution — open it via
*File → Open notebook → GitHub*, or upload it. It needs an `HF_TOKEN` in Colab's
secrets and accepted licences for the two gated models (Llama 3.2 and Gemma 3);
the other three run without a token. The notebook's first cell walks through both.

**Python → Rust** (`notebooks/python_to_rust.ipynb`). Claude Opus 5
(`anthropic/claude-opus-5`) and GPT-5 (`openai/gpt-5`), both through OpenRouter, are
asked to rewrite a Python program as a single-file Rust program with identical
output. Each port is streamed into `rust_build/main.rs`, compiled with `rustc
-Copt-level=3 -Ctarget-cpu=native`, and run three times, so the comparison is a
measured speedup rather than a plausible-looking diff. The worked example is a
200M-term series for π; the program is just a string, so any self-contained Python
that prints deterministic output can take its place.

Unlike the tour above this one runs **locally**, and it needs two things Colab would
have provided: an `OPENROUTER_API_KEY` in `.env`, and a Rust toolchain on your `PATH`
(install from [rustup.rs](https://rustup.rs)). Generated Rust lands in `rust_build/`,
which is gitignored.

```bash
uv run jupyter lab notebooks/python_to_rust.ipynb
```

## Development

Formatting, linting, and type checking:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy main.py
```

JupyterLab is a dev dependency, so the notebooks that run locally need no separate
install:

```bash
uv run jupyter lab
```

A pre-commit hook formats staged Python files with ruff. Enable it once per clone:

```bash
uv run pre-commit install
```