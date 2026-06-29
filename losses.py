"""Chunked cross-entropy for large-vocab Jefqwen models."""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LM_HEAD_CHUNK = 512


def _lm_head_device(model) -> torch.device:
    return next(model.lm_head.parameters()).device


@torch.no_grad()
def compute_batch_losses(
    model,
    token_batches: list[list[int]],
    device: str,
    chunk_size: int = DEFAULT_LM_HEAD_CHUNK,
) -> list[float]:
    """Forward pass with chunked lm_head to avoid OOM on large vocabs."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    if hasattr(model, "reset_state"):
        model.reset_state()
    hidden = model.model(input_ids).last_hidden_state
    lm_head = model.lm_head
    head_dev = _lm_head_device(model)
    if hidden.device != head_dev:
        hidden = hidden.to(head_dev)
    labels = input_ids if input_ids.device == head_dev else input_ids.to(head_dev)

    n_positions = labels.size(1) - 1
    total_loss = torch.zeros(len(token_batches), device=head_dev)

    for i in range(0, n_positions, chunk_size):
        end_pos = min(i + chunk_size, n_positions)
        chunk_logits = lm_head(hidden[:, i:end_pos, :])
        chunk_labels = labels[:, i + 1 : end_pos + 1]
        loss = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_labels.reshape(-1),
            reduction="none",
        )
        total_loss += loss.reshape(len(token_batches), -1).sum(dim=1)
        del chunk_logits, loss

    return (total_loss / n_positions).cpu().tolist()
