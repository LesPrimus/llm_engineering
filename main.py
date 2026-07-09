from dotenv import load_dotenv

from claude import ClaudeClient
from openrouter import OpenRouterClient

load_dotenv()

openrouter = OpenRouterClient()
claude = ClaudeClient()

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
]


def main() -> None:
    prompt = "Explain LoRA in one sentence."
    for model in MODELS:
        print(f"--- {model} ---")
        print(openrouter.ask(model, prompt))
        print()

    model = "meta-llama/llama-3.3-70b-instruct"
    print(f"--- {model} (pinned to Groq) ---")
    print(openrouter.ask(model, prompt, provider="groq"))
    print()

    model = "claude-opus-4-8"
    print(f"--- {model} (direct Anthropic API) ---")
    print(claude.ask(model, prompt))


if __name__ == "__main__":
    main()
