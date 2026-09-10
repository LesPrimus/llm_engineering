"""Measure how well a trained model does on data it is shown.

Training reports loss, which is the quantity the optimizer minimises rather
than the one anybody cares about — a loss of 0.03 says nothing legible about
whether the predictions are right. Accuracy converts the model's logits back
into decisions and counts them, which is the number you actually report.

The helper here takes a dataloader rather than raw tensors so the same call
works for the training set and the held-out set, and so the batching stays
wherever it was configured.
"""

import torch
from torch.utils.data import DataLoader


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader) -> float:
    """Return the fraction of examples in ``dataloader`` that ``model`` gets right.

    Runs one pass over ``dataloader``, takes the highest-scoring class per row
    (``argmax`` over the logits) and compares it against the label. Gradients
    are never recorded, so this is inference-only and costs no autograd memory.

    ``model`` is switched to evaluation mode for the pass and restored to
    whichever mode it arrived in, which makes the call safe from inside a
    training loop: a per-epoch accuracy readout will not silently leave the
    model in ``eval`` and disable dropout for every epoch that follows.

    An empty ``dataloader`` returns ``0.0`` rather than raising, so an
    accidentally-empty eval split shows up as a bad number instead of a crash.
    """
    was_training = model.training
    model.eval()

    correct = 0
    total_examples = 0

    try:
        with torch.no_grad():
            for features, labels in dataloader:
                logits = model(features)
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum())
                total_examples += labels.shape[0]
    finally:
        model.train(was_training)

    if total_examples == 0:
        return 0.0
    return correct / total_examples
