import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
]


def ask(model: str, prompt: str, provider: str | None = None) -> str:
    extra_body = {}
    if provider is not None:
        # Pin the request to one OpenRouter provider instead of letting it
        # pick the host by price/uptime; fail rather than fall back elsewhere.
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    response = client.chat.completions.create(
        model=model,
        messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    prompt = "Explain LoRA in one sentence."
    for model in MODELS:
        print(f"--- {model} ---")
        print(ask(model, prompt))
        print()

    model = "meta-llama/llama-3.3-70b-instruct"
    print(f"--- {model} (pinned to Groq) ---")
    print(ask(model, prompt, provider="groq"))


if __name__ == "__main__":
    main()
