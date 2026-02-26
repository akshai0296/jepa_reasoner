"""
Latent Graph Search & Refinement module.

Implements techniques for exploring and refining the latent reasoning space:
  - Latent graph search (BFS/beam search over latent states)
  - Denoising & perturbation for robustness
  - Iterative latent refinement via gradient-based optimization
  - Clustering of solution attempts for meta-critique
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentNode:
    """A node in the latent reasoning graph."""

    latent: torch.Tensor
    depth: int = 0
    score: float = 0.0
    parent_idx: int = -1
    decoded_text: str = ""


class LatentGraphSearch:
    """
    Explores a graph of latent reasoning states via beam search.

    Starting from the context embedding, the predictor generates multiple
    candidate next-step latents. Each can be decoded and scored by the
    verifier, then fed back for further prediction steps.
    """

    def __init__(
        self,
        predictor: nn.Module,
        decoder: nn.Module,
        verifier: Optional[nn.Module] = None,
        beam_width: int = 4,
        max_depth: int = 3,
        noise_scale: float = 0.1,
        num_perturbations: int = 4,
    ):
        self.predictor = predictor
        self.decoder = decoder
        self.verifier = verifier
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.noise_scale = noise_scale
        self.num_perturbations = num_perturbations

    @torch.no_grad()
    def search(
        self,
        s_x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> list[LatentNode]:
        """
        Beam search over latent states.

        Returns list of LatentNode results sorted by score (best first).
        """
        device = s_x.device

        initial_pred = self.predictor(s_x, z)
        candidates = []
        for i in range(self.num_perturbations):
            if i == 0:
                noisy = initial_pred
            else:
                noise = torch.randn_like(initial_pred) * self.noise_scale
                noisy = initial_pred + noise
            candidates.append(
                LatentNode(latent=noisy, depth=0, parent_idx=-1)
            )

        all_results = []
        beam = candidates

        for depth in range(self.max_depth):
            scored_beam = []
            for node in beam:
                texts = self.decoder.generate(node.latent, max_new_tokens=128, temperature=0.7)
                node.decoded_text = texts[0] if texts else ""

                if self.verifier is not None:
                    score = self.verifier(s_x, node.latent).mean().item()
                else:
                    score = -torch.norm(node.latent).item()
                node.score = score
                scored_beam.append(node)

            scored_beam.sort(key=lambda n: n.score, reverse=True)
            beam = scored_beam[: self.beam_width]
            all_results.extend(beam)

            if depth < self.max_depth - 1:
                next_beam = []
                for node in beam:
                    refined = self.predictor(node.latent.unsqueeze(0) if node.latent.dim() == 1 else node.latent, z)
                    for j in range(self.num_perturbations // 2 + 1):
                        if j == 0:
                            child_latent = refined
                        else:
                            noise = torch.randn_like(refined) * self.noise_scale * (0.5 ** depth)
                            child_latent = refined + noise
                        next_beam.append(
                            LatentNode(
                                latent=child_latent.squeeze(0) if child_latent.dim() > 2 else child_latent,
                                depth=depth + 1,
                                parent_idx=beam.index(node),
                            )
                        )
                beam = next_beam

        all_results.sort(key=lambda n: n.score, reverse=True)
        return all_results


class LatentRefiner:
    """
    Iteratively refines a latent prediction via gradient-based optimization
    against a verifier or energy function.
    """

    def __init__(
        self,
        verifier: nn.Module,
        num_steps: int = 20,
        lr: float = 0.01,
        noise_anneal: float = 0.95,
    ):
        self.verifier = verifier
        self.num_steps = num_steps
        self.lr = lr
        self.noise_anneal = noise_anneal

    def refine(
        self,
        s_x: torch.Tensor,
        s_y_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize s_y to maximize verifier score via gradient ascent.

        Args:
            s_x: context embedding (frozen)
            s_y_init: initial predicted latent

        Returns:
            refined latent vector
        """
        s_y = s_y_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([s_y], lr=self.lr)

        best_score = float("-inf")
        best_latent = s_y_init.detach().clone()

        for step in range(self.num_steps):
            optimizer.zero_grad()
            score = self.verifier(s_x, s_y)
            loss = -score.mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current_score = score.mean().item()
                if current_score > best_score:
                    best_score = current_score
                    best_latent = s_y.detach().clone()

        return best_latent


class LatentDenoiser:
    """
    Iterative denoising refinement: applies the predictor multiple times,
    treating its output as context for the next step, converging toward
    a stable latent representation.
    """

    def __init__(
        self,
        predictor: nn.Module,
        num_steps: int = 5,
        momentum: float = 0.5,
    ):
        self.predictor = predictor
        self.num_steps = num_steps
        self.momentum = momentum

    @torch.no_grad()
    def denoise(
        self,
        s_y_noisy: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Iteratively denoise a latent by re-predicting."""
        current = s_y_noisy

        for step in range(self.num_steps):
            refined = self.predictor(current, z)
            current = self.momentum * current + (1 - self.momentum) * refined

        return current


def cluster_latents(
    latents: list[torch.Tensor],
    n_clusters: int = 3,
) -> list[int]:
    """
    Cluster latent vectors to group similar solution attempts.
    Returns cluster assignment for each latent.
    """
    stacked = torch.stack([l.squeeze() if l.dim() > 1 else l for l in latents])

    if stacked.size(0) <= n_clusters:
        return list(range(stacked.size(0)))

    stacked_np = stacked.cpu().numpy()
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(stacked_np)
        return labels.tolist()
    except ImportError:
        # Fallback: simple distance-based assignment
        centroids = stacked_np[:n_clusters]
        labels = []
        for vec in stacked_np:
            dists = [((vec - c) ** 2).sum() for c in centroids]
            labels.append(min(range(len(dists)), key=lambda i: dists[i]))
        return labels
