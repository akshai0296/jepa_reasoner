"""
V2 Model: Improved JEPA Reasoner with joint encoding, chunked prediction,
and pretrained decoder.

Key improvements over v1:
  1. Joint [Q][SEP][A] encoding — encoder sees the full Q→A relationship
  2. Chunked latent prediction — predicts a sequence of latent tokens, not one vector
  3. Pretrained decoder — TinyLlama/Phi-2 conditioned on latent chunks via prefix
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders_v2 import JointSequenceEncoder, JointTargetEncoder
from .predictors_v2 import ChunkedLatentPredictor
from .decoders_v2 import ChunkedPretrainedDecoder, ChunkedSmallDecoder


class ImprovedJEPAReasoner(nn.Module):
    """
    V2 JEPA Reasoner.

    Training pipeline:
      1. Joint encoder processes "[Q] [SEP] [A]" → s_x + s_y_chunks (target)
      2. Predictor: s_x → predicted_chunks
      3. Loss: MSE(predicted_chunks, s_y_chunks) + contrastive
      4. Decoder: predicted_chunks → solution text

    Inference pipeline:
      1. Joint encoder processes "[Q]" only → s_x
      2. Predictor: s_x → predicted_chunks
      3. Decoder: predicted_chunks → solution text
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        decoder_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        latent_dim: int = 768,
        num_chunks: int = 8,
        predictor_layers: int = 6,
        predictor_heads: int = 8,
        predictor_hidden: int = 1024,
        freeze_encoder: bool = False,
        freeze_decoder: bool = True,
        use_pretrained_decoder: bool = True,
        contrastive_temp: float = 0.07,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_chunks = num_chunks
        self.contrastive_temp = contrastive_temp

        self.encoder = JointSequenceEncoder(
            model_name=encoder_model,
            latent_dim=latent_dim,
            num_latent_chunks=num_chunks,
            freeze_backbone=freeze_encoder,
        )

        self.target_encoder = JointTargetEncoder(self.encoder, ema_decay=ema_decay)

        self.predictor = ChunkedLatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            num_chunks=num_chunks,
        )

        if use_pretrained_decoder:
            self.decoder = ChunkedPretrainedDecoder(
                model_name=decoder_model,
                latent_dim=latent_dim,
                num_chunks=num_chunks,
                freeze_backbone=freeze_decoder,
            )
        else:
            self.decoder = ChunkedSmallDecoder(
                latent_dim=latent_dim,
                num_chunks=num_chunks,
            )

    def forward(
        self,
        questions: list[str],
        answers: list[str],
        solution_ids: Optional[torch.Tensor] = None,
        solution_mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        pred_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        decoder_weight: float = 0.5,
    ) -> dict:
        """
        Training forward pass.

        Args:
            questions: list of problem strings
            answers:   list of solution strings
            solution_ids/mask: tokenized solutions for decoder training
        """
        device = next(self.parameters()).device

        # Joint encode [Q][SEP][A] → s_x + s_y_chunks (target)
        joint_tokens = self.encoder.tokenize_joint(questions, answers)
        joint_tokens = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in joint_tokens.items()}

        enc_out = self.encoder(
            joint_tokens["input_ids"],
            joint_tokens["attention_mask"],
            joint_tokens["answer_starts"],
        )
        s_x = enc_out["s_x"]
        s_y_chunks = enc_out["s_y_chunks"]  # [B, K, latent_dim]

        # Also get EMA target
        with torch.no_grad():
            target_out = self.target_encoder(
                joint_tokens["input_ids"],
                joint_tokens["attention_mask"],
                joint_tokens["answer_starts"],
            )
            s_y_target = target_out["s_y_chunks"]  # [B, K, latent_dim]

        # Predict chunks from context
        pred_chunks = self.predictor(s_x, z)  # [B, K, latent_dim]

        # Chunked prediction loss: MSE per chunk
        pred_loss = F.mse_loss(pred_chunks, s_y_target)

        # Contrastive: flatten chunks to single vector for InfoNCE
        pred_flat = pred_chunks.mean(dim=1)  # [B, latent_dim]
        target_flat = s_y_target.mean(dim=1)
        c_loss = self._contrastive_loss(pred_flat, target_flat)

        total_loss = pred_weight * pred_loss + contrastive_weight * c_loss

        # Decoder loss
        dec_loss = torch.tensor(0.0, device=device)
        if solution_ids is not None:
            dec_out = self.decoder(pred_chunks, solution_ids, solution_mask)
            dec_loss = dec_out["loss"]
            total_loss = total_loss + decoder_weight * dec_loss

        # EMA update
        self.target_encoder.ema_update(self.encoder)

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "contrastive_loss": c_loss,
            "decoder_loss": dec_loss,
            "s_x": s_x.detach(),
            "pred_chunks": pred_chunks.detach(),
            "target_chunks": s_y_target.detach(),
        }

    def _contrastive_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        logits = torch.matmul(pred_n, target_n.T) / self.contrastive_temp
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    @torch.no_grad()
    def inference(
        self,
        questions: list[str],
        z: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict:
        """Inference: encode question → predict chunks → decode to text."""
        self.eval()
        device = next(self.parameters()).device

        tokens = self.encoder.tokenize_context_only(questions)
        tokens = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

        enc_out = self.encoder(tokens["input_ids"], tokens["attention_mask"])
        s_x = enc_out["s_x"]

        pred_chunks = self.predictor(s_x, z)  # [B, K, latent_dim]

        texts = self.decoder.generate(
            pred_chunks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return {
            "texts": texts,
            "s_x": s_x,
            "pred_chunks": pred_chunks,
        }
