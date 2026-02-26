"""
Evaluation metrics for the I-JEPA reasoning system.

Covers latent space quality, task accuracy, and solution quality.
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np


def embedding_distance(s_y: torch.Tensor, s_y_hat: torch.Tensor, metric: str = "l2") -> float:
    """Average distance between predicted and actual target embeddings."""
    if metric == "l2":
        return F.mse_loss(s_y_hat, s_y).item()
    elif metric == "cosine":
        return (1 - F.cosine_similarity(s_y_hat, s_y, dim=-1)).mean().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def cosine_similarity_score(s_y: torch.Tensor, s_y_hat: torch.Tensor) -> float:
    """Average cosine similarity between predicted and actual embeddings."""
    return F.cosine_similarity(s_y_hat, s_y, dim=-1).mean().item()


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract a numeric final answer from solution text."""
    patterns = [
        r"####\s*(-?[\d,.]+)",
        r"(?:answer|result|solution)\s*(?:is|=|:)\s*(-?[\d,.]+)",
        r"=\s*(-?[\d,.]+)\s*$",
        r"(-?[\d,.]+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def exact_match_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Fraction of predictions that exactly match the target (after strip/lower)."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, t in zip(predictions, targets) if p.strip().lower() == t.strip().lower()
    )
    return correct / len(predictions)


def numeric_accuracy(predictions: list[str], targets: list[str], tolerance: float = 1e-3) -> float:
    """Fraction of predictions with numerically correct final answer."""
    if not predictions:
        return 0.0
    correct = 0
    evaluated = 0
    for pred, target in zip(predictions, targets):
        pred_num = extract_numeric_answer(pred)
        target_num = extract_numeric_answer(target)
        if target_num is not None:
            evaluated += 1
            if pred_num is not None and abs(pred_num - target_num) <= tolerance:
                correct += 1
    return correct / max(evaluated, 1)


def latent_space_stats(embeddings: torch.Tensor) -> dict:
    """Compute statistics about the embedding space to detect collapse."""
    norms = torch.norm(embeddings, dim=-1)
    mean_embedding = embeddings.mean(dim=0)
    centered = embeddings - mean_embedding
    variance = (centered ** 2).mean(dim=0).sum().item()

    pairwise_cos = F.cosine_similarity(
        embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1
    )
    mask = ~torch.eye(pairwise_cos.size(0), dtype=torch.bool, device=pairwise_cos.device)
    avg_pairwise_cos = pairwise_cos[mask].mean().item()

    return {
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "variance": variance,
        "avg_pairwise_cosine": avg_pairwise_cos,
        "collapse_detected": avg_pairwise_cos > 0.95,
    }


def compute_silhouette(embeddings: torch.Tensor, labels: list[int]) -> float:
    """Compute silhouette score for clustering quality of latent space."""
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(embeddings.cpu().numpy(), labels)
    except (ImportError, ValueError):
        return 0.0


def rank_candidates_by_similarity(
    target_embedding: torch.Tensor, candidate_embeddings: list[torch.Tensor]
) -> list[int]:
    """Rank candidates by cosine similarity to target (best first)."""
    scores = []
    for cand in candidate_embeddings:
        sim = F.cosine_similarity(target_embedding, cand, dim=-1).mean().item()
        scores.append(sim)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class MetricsTracker:
    """Accumulates and reports training/evaluation metrics."""

    def __init__(self):
        self.history = {}

    def update(self, metrics: dict, step: int):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.history.setdefault(key, []).append((step, value))

    def get_latest(self, key: str) -> Optional[float]:
        if key in self.history and self.history[key]:
            return self.history[key][-1][1]
        return None

    def get_average(self, key: str, window: int = 10) -> Optional[float]:
        if key not in self.history:
            return None
        values = [v for _, v in self.history[key][-window:]]
        return sum(values) / len(values) if values else None

    def summary(self) -> dict:
        return {key: self.get_latest(key) for key in self.history}

    def to_dict(self) -> dict:
        return dict(self.history)
