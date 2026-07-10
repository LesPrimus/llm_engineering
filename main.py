from dotenv import load_dotenv

from openrouter import run_conversation

load_dotenv()


def main() -> None:
    run_conversation(rounds=5)


if __name__ == "__main__":
    main()
