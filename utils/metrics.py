from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def multiclass_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def multiclass_macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    if logits.numel() == 0 or labels.numel() == 0 or num_classes <= 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    f1_values = []
    for class_id in range(num_classes):
        pred_pos = predictions == class_id
        label_pos = labels == class_id
        tp = (pred_pos & label_pos).sum().float()
        fp = (pred_pos & ~label_pos).sum().float()
        fn = (~pred_pos & label_pos).sum().float()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
        f1_values.append(f1)
    return float(torch.stack(f1_values).mean().item())


def summarize_scores(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def hit_rate_at_k(ranked_items: Sequence[int], positive_item: int, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(positive_item in list(ranked_items)[:k])


def ndcg_at_k(ranked_items: Sequence[int], positive_item: int, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(ranked_items)[:k]
    if positive_item not in top_k:
        return 0.0
    rank = top_k.index(positive_item) + 1
    return float(1.0 / torch.log2(torch.tensor(float(rank + 1))).item())
