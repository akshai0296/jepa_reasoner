"""
Domain-specific context and target encoders for the I-JEPA reasoning system.

Each domain (code, math, text) has a context encoder and a target encoder.
Context encoders process the problem/question; target encoders process the
solution/answer. Both produce fixed-size latent vectors in a shared embedding space.

The target encoder can optionally use EMA (exponential moving average) of the
context encoder weights, following I-JEPA's stabilization strategy.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BaseEncoder(nn.Module, ABC):
    """Base encoder that maps tokenized input to a fixed-size latent vector."""

    def __init__(self, latent_dim: int = 768):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns a [batch, latent_dim] embedding."""
        ...


class TransformerEncoder(BaseEncoder):
    """Wraps a HuggingFace transformer backbone and projects to latent_dim."""

    def __init__(
        self,
        model_name: str,
        latent_dim: int = 768,
        pooling: str = "cls",
        freeze_backbone: bool = False,
    ):
        super().__init__(latent_dim)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

        backbone_dim = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return self.projection(pooled)


class CodeContextEncoder(TransformerEncoder):
    """Context encoder for the coding domain, initialized from CodeBERT."""

    def __init__(self, latent_dim: int = 768, freeze_backbone: bool = False):
        super().__init__(
            model_name="microsoft/codebert-base",
            latent_dim=latent_dim,
            pooling="cls",
            freeze_backbone=freeze_backbone,
        )


class MathContextEncoder(TransformerEncoder):
    """Context encoder for the math domain, initialized from a sentence-transformer."""

    def __init__(self, latent_dim: int = 768, freeze_backbone: bool = False):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            latent_dim=latent_dim,
            pooling="mean",
            freeze_backbone=freeze_backbone,
        )


class TextContextEncoder(TransformerEncoder):
    """Context encoder for text reasoning, initialized from a sentence-transformer."""

    def __init__(self, latent_dim: int = 768, freeze_backbone: bool = False):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            latent_dim=latent_dim,
            pooling="mean",
            freeze_backbone=freeze_backbone,
        )


class TargetEncoder(nn.Module):
    """
    Target encoder that mirrors a context encoder's architecture.

    Supports EMA (exponential moving average) updates from the context encoder,
    which stabilizes training targets per I-JEPA methodology.
    """

    def __init__(self, context_encoder: TransformerEncoder, ema_decay: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(context_encoder)
        self.ema_decay = ema_decay

        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def ema_update(self, context_encoder: TransformerEncoder):
        """Update target encoder weights as EMA of context encoder."""
        for target_param, context_param in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(self.ema_decay).add_(
                context_param.data, alpha=1.0 - self.ema_decay
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(input_ids, attention_mask)


ENCODER_REGISTRY = {
    "code": CodeContextEncoder,
    "math": MathContextEncoder,
    "text": TextContextEncoder,
}


def build_encoders(
    domain: str,
    latent_dim: int = 768,
    freeze_backbone: bool = False,
    ema_decay: float = 0.996,
) -> tuple[TransformerEncoder, TargetEncoder]:
    """Factory: builds a (context_encoder, target_encoder) pair for a domain."""
    if domain not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {list(ENCODER_REGISTRY)}")

    context_enc = ENCODER_REGISTRY[domain](latent_dim=latent_dim, freeze_backbone=freeze_backbone)
    target_enc = TargetEncoder(context_enc, ema_decay=ema_decay)
    return context_enc, target_enc
