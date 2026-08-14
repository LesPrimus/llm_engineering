"""
Async version of the Week 5 'pro implementation' ingestion pipeline.

Requires Python 3.12+ for itertools.batched.
"""

import asyncio
from itertools import batched
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import acompletion
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100

wait = wait_exponential(multiplier=1, min=10, max=240)

CHUNK_CONCURRENCY = 3
EMBED_CONCURRENCY = 4
EMBED_BATCH = 200

async_openai = AsyncOpenAI()


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, "
        "that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk "
        "to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, "
        "exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"type": doc_type, "source": file.as_posix(), "text": f.read()}
                )
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""You take a document and you split the document into overlapping chunks for a KnowledgeBase.
The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}
A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.
For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [{"role": "user", "content": make_prompt(document)}]


# --------------------------------------------------------------------------
# Stage 1: chunking
# --------------------------------------------------------------------------


@retry(wait=wait)
async def process_document(document, semaphore):
    async with semaphore:
        response = await acompletion(
            model=MODEL, messages=make_messages(document), response_format=Chunks
        )
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


async def create_chunks(documents):
    """Chunk every document, at most CHUNK_CONCURRENCY requests in flight."""
    semaphore = asyncio.Semaphore(CHUNK_CONCURRENCY)
    results = await tqdm_asyncio.gather(
        *(process_document(d, semaphore) for d in documents),
        desc="Chunking",
    )
    return [chunk for result in results for chunk in result]


# --------------------------------------------------------------------------
# Stage 2: embedding + storage
# --------------------------------------------------------------------------


@retry(wait=wait)
async def embed_batch(texts):
    response = await async_openai.embeddings.create(model=embedding_model, input=texts)
    return [e.embedding for e in response.data]


async def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    collection = chroma.get_or_create_collection(collection_name)

    batches = list(batched(chunks, EMBED_BATCH))
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
    write_lock = asyncio.Lock()

    async def handle(batch_index, batch):
        texts = [chunk.page_content for chunk in batch]
        async with semaphore:
            vectors = await embed_batch(texts)

        offset = batch_index * EMBED_BATCH
        ids = [str(offset + i) for i in range(len(batch))]
        metas = [chunk.metadata for chunk in batch]

        # collection.add is sync and not concurrency-safe: serialise it,
        # and run it off the event loop so it doesn't block other batches.
        async with write_lock:
            await asyncio.to_thread(
                collection.add,
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metas,
            )

    await tqdm_asyncio.gather(
        *(handle(i, b) for i, b in enumerate(batches)),
        desc="Embedding",
    )
    print(f"Vectorstore created with {collection.count()} documents")


# --------------------------------------------------------------------------


async def main():
    documents = fetch_documents()
    chunks = await create_chunks(documents)
    await create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    asyncio.run(main())
