"""
V2 Predictor: Chunked latent prediction.

Instead of predicting a single latent vector, predicts a SEQUENCE of
latent chunks. Each chunk encodes a portion of the solution's semantics.
This gives the decoder much more information to work with.

Architecture:
  s_x [B, latent_dim] → project + add learnable chunk queries
  → transformer layers → [B, num_chunks, latent_dim]
"""

from typing import Optional

import torch
import torch.nn as nn


class ChunkedLatentPredictor(nn.Module):
    """
    Predicts a sequence of latent chunks from context embedding.

    Instead of: s_x → one vector ŝ_y
    Now:        s_x → [ŝ_y_1, ŝ_y_2, ..., ŝ_y_K]  (K chunks)

    Each chunk captures a different aspect/step of the solution.
    The decoder can attend to all chunks, getting far richer signal.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        num_chunks: int = 8,
        dropout: float = 0.1,
        use_latent_z: bool = False,
        z_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_chunks = num_chunks
        self.use_latent_z = use_latent_z
        self.z_dim = z_dim

        input_dim = latent_dim + z_dim if use_latent_z else latent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable chunk queries that the transformer refines
        self.chunk_queries = nn.Parameter(torch.randn(1, num_chunks, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_chunks + 1, hidden_dim) * 0.02)

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
            predicted chunks [batch, num_chunks, latent_dim]
        """
        batch_size = s_x.size(0)

        if self.use_latent_z:
            if z is None:
                z = torch.zeros(batch_size, self.z_dim, device=s_x.device)
            x = torch.cat([s_x, z], dim=-1)
        else:
            x = s_x

        x = self.input_proj(x).unsqueeze(1)  # [B, 1, hidden]
        queries = self.chunk_queries.expand(batch_size, -1, -1)
        seq = torch.cat([x, queries], dim=1)  # [B, 1+K, hidden]
        seq = seq + self.pos_embed[:, :seq.size(1)]

        seq = self.transformer(seq)
        seq = self.norm(seq)

        chunks = seq[:, 1:]  # [B, K, hidden] — skip the context token
        return self.output_proj(chunks)  # [B, K, latent_dim]
