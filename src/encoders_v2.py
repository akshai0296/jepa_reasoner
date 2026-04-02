"""
V2 Encoders: Joint sequence encoding for the improved JEPA reasoner.

Key change from v1: instead of encoding Q and A separately, we encode
"[Q] [SEP] [A]" as a single sequence. The encoder sees the full context
of how the answer relates to the question, producing richer latents.

During training:  encode("[Q] [SEP] [A]") → extract answer-region latents → s_y
During inference: encode("[Q]") → s_x, then predictor fills in the answer latents
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class JointSequenceEncoder(nn.Module):
    """
    Encodes "[question] [SEP] [answer]" as a single sequence.

    Produces two outputs:
      - s_x: latent from the question region (always available)
      - s_y: latent from the answer region (only during training)

    The answer portion's representation is richer because the transformer
    attends across both Q and A, capturing the relationship.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        latent_dim: int = 768,
        num_latent_chunks: int = 8,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.latent_dim = latent_dim
        self.num_latent_chunks = num_latent_chunks

        backbone_dim = self.backbone.config.hidden_size

        self.context_proj = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Projects answer-region hidden states into a SEQUENCE of latent chunks
        self.answer_proj = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Cross-attention to compress variable-length answer hiddens into fixed chunks
        self.chunk_queries = nn.Parameter(torch.randn(1, num_latent_chunks, latent_dim) * 0.02)
        self.chunk_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=8, batch_first=True
        )
        self.chunk_norm = nn.LayerNorm(latent_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def tokenize_joint(
        self,
        questions: list[str],
        answers: list[str],
        max_q_len: int = 128,
        max_a_len: int = 128,
    ) -> dict:
        """Tokenize "[Q] [SEP] [A]" and track where the answer starts."""
        sep_token = self.tokenizer.sep_token or " [SEP] "

        joint_texts = [f"{q} {sep_token} {a}" for q, a in zip(questions, answers)]

        q_lens = []
        for q in questions:
            q_enc = self.tokenizer(q, add_special_tokens=False)
            q_lens.append(min(len(q_enc["input_ids"]), max_q_len))

        encoding = self.tokenizer(
            joint_texts,
            max_length=max_q_len + max_a_len + 3,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # +1 for CLS, +1 for SEP token in the middle
        answer_starts = [ql + 2 for ql in q_lens]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "answer_starts": answer_starts,
        }

    def tokenize_context_only(self, questions: list[str], max_len: int = 128) -> dict:
        """Tokenize just the question (for inference)."""
        encoding = self.tokenizer(
            questions,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_starts: Optional[list[int]] = None,
    ) -> dict:
        """
        Returns:
            s_x: [batch, latent_dim] context embedding (mean pool over Q region)
            s_y_chunks: [batch, num_chunks, latent_dim] answer latent chunks
                        (None if answer_starts not provided, i.e. inference mode)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq, backbone_dim]

        batch_size = hidden.size(0)

        if answer_starts is not None:
            # Training: split hidden states into Q and A regions
            q_embeds = []
            a_embeds_list = []

            for i in range(batch_size):
                sep_pos = answer_starts[i]
                q_hidden = hidden[i, 1:sep_pos]  # skip CLS
                a_hidden = hidden[i, sep_pos:]

                q_mask = attention_mask[i, 1:sep_pos].float().unsqueeze(-1)
                q_pooled = (q_hidden * q_mask).sum(0) / q_mask.sum(0).clamp(min=1e-9)
                q_embeds.append(q_pooled)

                a_proj = self.answer_proj(a_hidden)
                a_mask = attention_mask[i, sep_pos:].float()
                valid_len = int(a_mask.sum().item())
                if valid_len == 0:
                    valid_len = 1
                a_embeds_list.append(a_proj[:valid_len])

            s_x = self.context_proj(torch.stack(q_embeds))

            # Compress answer embeddings into fixed-size chunks via cross-attention
            max_a_len = max(a.size(0) for a in a_embeds_list)
            a_padded = torch.zeros(batch_size, max_a_len, self.latent_dim, device=hidden.device)
            a_key_mask = torch.ones(batch_size, max_a_len, dtype=torch.bool, device=hidden.device)
            for i, a in enumerate(a_embeds_list):
                a_padded[i, :a.size(0)] = a
                a_key_mask[i, :a.size(0)] = False

            queries = self.chunk_queries.expand(batch_size, -1, -1)
            s_y_chunks, _ = self.chunk_attn(queries, a_padded, a_padded, key_padding_mask=a_key_mask)
            s_y_chunks = self.chunk_norm(s_y_chunks + queries)

            return {"s_x": s_x, "s_y_chunks": s_y_chunks}

        else:
            # Inference: only encode question, no answer region
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            s_x = self.context_proj(pooled)

            return {"s_x": s_x, "s_y_chunks": None}


class JointTargetEncoder(nn.Module):
    """EMA-updated copy of the JointSequenceEncoder for stable targets."""

    def __init__(self, encoder: JointSequenceEncoder, ema_decay: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.ema_decay = ema_decay
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def ema_update(self, source_encoder: JointSequenceEncoder):
        for tp, sp in zip(self.encoder.parameters(), source_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, answer_starts=None):
        return self.encoder(input_ids, attention_mask, answer_starts)
