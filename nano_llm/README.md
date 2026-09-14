# nano_llm

Building an LLM from scratch with plain PyTorch — no `transformers` or other
model libraries.

## Layout

- Modules live at the package root.
- Notebooks live in `notebooks/`, named `NN_topic.ipynb`.

## From text to input embeddings

The data path in front of the model, with the shapes from
`notebooks/text_to_embeddings.ipynb` (`batch_size=8`, `max_length=4`,
`output_dim=256`):

```
raw text            "I HAD always thought Jack Gisburn rather a cheap genius..."
  |  tiktoken "gpt2" BPE
token IDs           [40, 367, 2885, 1464, 1807, 3619, ...]        50,257 vocab
  |  GPTDatasetV1: sliding windows of max_length, step=stride
windows             ([40, 367, 2885, 1464], [367, 2885, 1464, 1807])   input, target
  |  DataLoader: stack batch_size windows
inputs              [8, 4]          int64 token IDs
  |  token_embedding_layer: row lookup in a [50257, 256] table
token embeddings    [8, 4, 256]     float vectors
  |  + pos_embeddings [4, 256], broadcast over the batch
input embeddings    [8, 4, 256]     what the model consumes
```

**Tokenization.** `tiktoken`'s GPT-2 encoding splits the text into byte-pair
tokens and maps each to an integer. The vocabulary is 50,000 learned merges
plus 256 raw byte tokens plus `<|endoftext|>`. The byte fallback means any
input encodes — there is no unknown token.

**Windows.** `GPTDatasetV1` slices the token stream into chunks of
`max_length`, advancing by `stride`. Each chunk's target is the same chunk
shifted one position right, which is the next-token prediction objective.
`stride == max_length` gives non-overlapping windows; a smaller stride
overlaps them and yields more training pairs from the same text.

**Batching.** `create_dataloader_v1` wraps the dataset in a `DataLoader` that
stacks `batch_size` windows into `[batch, seq]`. Still integers at this point.

**Token embeddings.** `torch.nn.Embedding(vocab_size, output_dim)` is a
`[50257, 256]` table of learned parameters. The token ID is a row index, so
the lookup swaps every integer for its 256-dim vector and the rank goes from
2 to 3. The IDs are arbitrary labels, not quantities — this is the step that
gives them meaning the network can do arithmetic with.

**Positional embeddings.** A second table, `[max_length, output_dim]`, indexed
by `torch.arange(max_length)` rather than by data. One row per slot in the
window, so it answers *where* a token sits while the first table answers
*what* it is. Position is absolute and counted from the start of the window:
the eighth row of the batch holds tokens 28-31 of the text but still gets
positions 0-3.

**The sum.** `[8, 4, 256] + [4, 256]` broadcasts — the missing batch axis is
treated as 1 and stretched to 8, so every sequence receives the same four
position vectors. The addition is elementwise, so the width stays 256; the
two meanings are summed into one vector rather than concatenated. The result
is the model's working shape, and every transformer block preserves it.

## What comes next

Not written yet, in the order they land:

- **Self-attention** — the first step where positions interact. Until now each
  token's vector is independent of its neighbours; attention mixes them, which
  is what makes the representations contextual.
- **Transformer block** — attention plus a feed-forward layer, wrapped in layer
  norms and residual connections. Takes `[batch, seq, 256]` and returns the
  same shape, which is what lets the blocks stack.
- **Output head** — a `[256, 50257]` projection turning each position's vector
  back into a score per vocabulary entry, trained against the shifted targets
  with cross-entropy.
- **Training loop** — the pattern already in `notebooks/llm_basic_training.ipynb`
  (`zero_grad`, `backward`, `step`), scaled up to this model.
