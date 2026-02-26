"""
Latent Predictor -- the I-JEPA core.

Takes a context embedding s_x (and optionally a latent variable z) and predicts
the target embedding s_y. This is the "world model" that performs reasoning in
latent space without generating tokens.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Transformer-based predictor that maps context embeddings to predicted
    target embeddings. Optionally conditions on a latent variable z for
    multi-modal / uncertain predictions.

    Architecture: projects input to a sequence of tokens, runs through
    transformer layers, then projects back to latent_dim.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_latent_z: bool = False,
        z_dim: int = 64,
        num_predictor_tokens: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_latent_z = use_latent_z
        self.z_dim = z_dim
        self.num_predictor_tokens = num_predictor_tokens

        input_dim = latent_dim + z_dim if use_latent_z else latent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.predictor_tokens = nn.Parameter(
            torch.randn(1, num_predictor_tokens, hidden_dim) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_predictor_tokens + 1, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        s_x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s_x: context embedding [batch, latent_dim]
            z:   optional latent variable [batch, z_dim]

        Returns:
            s_y_pred: predicted target embedding [batch, latent_dim]
        """
        batch_size = s_x.size(0)

        if self.use_latent_z:
            if z is None:
                z = torch.zeros(batch_size, self.z_dim, device=s_x.device)
            x = torch.cat([s_x, z], dim=-1)
        else:
            x = s_x

        x = self.input_proj(x).unsqueeze(1)  # [B, 1, hidden]

        tokens = self.predictor_tokens.expand(batch_size, -1, -1)
        seq = torch.cat([x, tokens], dim=1)  # [B, 1+N, hidden]
        seq = seq + self.pos_embed[:, : seq.size(1)]

        seq = self.transformer(seq)
        seq = self.norm(seq)

        # Pool over predictor tokens (skip the input token at position 0)
        pred_tokens = seq[:, 1:]  # [B, N, hidden]
        pooled = pred_tokens.mean(dim=1)  # [B, hidden]

        return self.output_proj(pooled)


class MLPPredictor(nn.Module):
    """Simpler MLP-based predictor as a lightweight alternative."""

    def __init__(
        self,
        latent_dim: int = 768,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_latent_z: bool = False,
        z_dim: int = 64,
    ):
        super().__init__()
        self.use_latent_z = use_latent_z
        self.z_dim = z_dim

        input_dim = latent_dim + z_dim if use_latent_z else latent_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [latent_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        s_x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_latent_z:
            if z is None:
                z = torch.zeros(s_x.size(0), self.z_dim, device=s_x.device)
            x = torch.cat([s_x, z], dim=-1)
        else:
            x = s_x
        return self.net(x)


def build_predictor(
    predictor_type: str = "transformer",
    latent_dim: int = 768,
    **kwargs,
) -> nn.Module:
    if predictor_type == "transformer":
        return LatentPredictor(latent_dim=latent_dim, **kwargs)
    elif predictor_type == "mlp":
        return MLPPredictor(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
