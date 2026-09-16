import asyncio
from litellm import acompletion


async def call_llm(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """LLM call with rate limiting and automatic retry."""
    async with semaphore:
        response = await acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            num_retries=3,  # Automatic retry with exponential backoff
        )
        return response.choices[0].message.content


async def main():
    semaphore = asyncio.Semaphore(3)
    prompts = [f"What is {i} + {i}?" for i in range(10)]

    results = await asyncio.gather(
        *(call_llm(prompt, semaphore) for prompt in prompts),
        return_exceptions=True,  # one failure doesn't cancel the rest
    )

    for prompt, result in zip(prompts, results):
        if isinstance(result, Exception):
            print(f"{prompt} -> failed: {result!r}")
        else:
            print(f"{prompt} -> {result}")


if __name__ == "__main__":
    asyncio.run(main())
