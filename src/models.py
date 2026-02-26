"""
High-level JEPAReasoner model that ties together encoders, predictor, and decoder
into a single cohesive system for training and inference.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TransformerEncoder, TargetEncoder, build_encoders
from .predictors import build_predictor
from .decoders import build_decoder


class JEPAReasoner(nn.Module):
    """
    End-to-end I-JEPA reasoning system.

    Pipeline: x -> ContextEncoder -> s_x -> Predictor(s_x, z) -> s_y_hat -> Decoder -> y_hat

    Training objectives:
      1. Latent prediction loss: D(s_y, s_y_hat) where s_y = TargetEncoder(y)
      2. Contrastive loss: InfoNCE to prevent collapse
      3. Decoder loss: cross-entropy on solution tokens (conditioned on s_y_hat)
    """

    def __init__(
        self,
        domain: str = "math",
        latent_dim: int = 768,
        predictor_type: str = "transformer",
        predictor_kwargs: Optional[dict] = None,
        decoder_type: str = "scratch",
        decoder_kwargs: Optional[dict] = None,
        freeze_backbone: bool = False,
        ema_decay: float = 0.996,
        contrastive_temp: float = 0.07,
    ):
        super().__init__()
        self.domain = domain
        self.latent_dim = latent_dim
        self.contrastive_temp = contrastive_temp

        self.context_encoder, self.target_encoder = build_encoders(
            domain=domain,
            latent_dim=latent_dim,
            freeze_backbone=freeze_backbone,
            ema_decay=ema_decay,
        )

        predictor_kwargs = predictor_kwargs or {}
        self.predictor = build_predictor(
            predictor_type=predictor_type,
            latent_dim=latent_dim,
            **predictor_kwargs,
        )

        decoder_kwargs = decoder_kwargs or {}
        decoder_kwargs.setdefault("latent_dim", latent_dim)
        self.decoder = build_decoder(decoder_type=decoder_type, **decoder_kwargs)

    def encode_context(self, input_ids, attention_mask):
        return self.context_encoder(input_ids, attention_mask)

    def encode_target(self, input_ids, attention_mask):
        return self.target_encoder(input_ids, attention_mask)

    def predict(self, s_x, z=None):
        return self.predictor(s_x, z)

    def decode(self, s_y_hat, target_ids=None, target_mask=None):
        if target_ids is not None:
            return self.decoder(s_y_hat, target_ids, target_mask)
        return self.decoder.generate(s_y_hat)

    def prediction_loss(self, s_y: torch.Tensor, s_y_hat: torch.Tensor, loss_type: str = "l2"):
        """Compute latent prediction loss D(s_y, s_y_hat)."""
        if loss_type == "l2":
            return F.mse_loss(s_y_hat, s_y)
        elif loss_type == "cosine":
            return 1.0 - F.cosine_similarity(s_y_hat, s_y, dim=-1).mean()
        elif loss_type == "smooth_l1":
            return F.smooth_l1_loss(s_y_hat, s_y)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def contrastive_loss(self, s_x_batch: torch.Tensor, s_y_batch: torch.Tensor):
        """
        InfoNCE contrastive loss.
        Pushes matching (s_x, s_y) pairs together, non-matching apart.
        """
        s_x_norm = F.normalize(s_x_batch, dim=-1)
        s_y_norm = F.normalize(s_y_batch, dim=-1)

        logits = torch.matmul(s_x_norm, s_y_norm.T) / self.contrastive_temp
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_xy = F.cross_entropy(logits, labels)
        loss_yx = F.cross_entropy(logits.T, labels)
        return (loss_xy + loss_yx) / 2

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        solution_ids: Optional[torch.Tensor] = None,
        solution_mask: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        loss_type: str = "l2",
        pred_loss_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        decoder_weight: float = 0.5,
    ) -> dict:
        """
        Full forward pass for training.

        Args:
            context_ids/mask: tokenized problem input
            target_ids/mask:  tokenized solution (for target encoder)
            solution_ids/mask: tokenized solution (for decoder training, may differ from target)
            z: optional latent variable
            loss_type: 'l2', 'cosine', or 'smooth_l1'

        Returns:
            dict with total_loss, pred_loss, contrastive_loss, decoder_loss, s_y_hat
        """
        s_x = self.encode_context(context_ids, context_mask)
        s_y = self.encode_target(target_ids, target_mask)
        s_y_hat = self.predict(s_x, z)

        pred_loss = self.prediction_loss(s_y, s_y_hat, loss_type)
        c_loss = self.contrastive_loss(s_y_hat, s_y)

        total_loss = pred_loss_weight * pred_loss + contrastive_weight * c_loss

        decoder_loss = torch.tensor(0.0, device=s_x.device)
        if solution_ids is not None:
            dec_out = self.decode(s_y_hat, solution_ids, solution_mask)
            decoder_loss = dec_out["loss"]
            total_loss = total_loss + decoder_weight * decoder_loss

        self.target_encoder.ema_update(self.context_encoder)

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "contrastive_loss": c_loss,
            "decoder_loss": decoder_loss,
            "s_x": s_x.detach(),
            "s_y": s_y.detach(),
            "s_y_hat": s_y_hat.detach(),
        }

    @torch.no_grad()
    def inference(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        num_candidates: int = 1,
    ) -> dict:
        """
        Inference pipeline: encode context -> predict latent -> decode to text.

        Args:
            num_candidates: generate multiple candidates by adding noise to z.
        """
        self.eval()
        s_x = self.encode_context(context_ids, context_mask)

        candidates = []
        latents = []

        for i in range(num_candidates):
            z_i = z
            if z_i is None and num_candidates > 1:
                z_noise = torch.randn(s_x.size(0), 64, device=s_x.device) * 0.5
                if hasattr(self.predictor, "use_latent_z") and self.predictor.use_latent_z:
                    z_i = z_noise

            s_y_hat = self.predict(s_x, z_i)

            if num_candidates > 1 and i > 0:
                noise = torch.randn_like(s_y_hat) * 0.1
                s_y_hat = s_y_hat + noise

            texts = self.decoder.generate(
                s_y_hat,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            candidates.append(texts)
            latents.append(s_y_hat)

        return {
            "candidates": candidates,
            "latents": latents,
            "s_x": s_x,
        }


class MultiDomainJEPAReasoner(nn.Module):
    """
    Wraps multiple single-domain JEPAReasoner instances, routing inputs
    to the appropriate domain model based on a domain identifier.
    """

    def __init__(self, domains: list[str], shared_predictor: bool = True, **kwargs):
        super().__init__()
        self.domains = domains
        self.models = nn.ModuleDict()

        for domain in domains:
            self.models[domain] = JEPAReasoner(domain=domain, **kwargs)

        if shared_predictor:
            shared = self.models[domains[0]].predictor
            for domain in domains[1:]:
                self.models[domain].predictor = shared

    def forward(self, domain: str, **kwargs):
        if domain not in self.models:
            raise ValueError(f"Unknown domain: {domain}")
        return self.models[domain](**kwargs)

    def inference(self, domain: str, **kwargs):
        if domain not in self.models:
            raise ValueError(f"Unknown domain: {domain}")
        return self.models[domain].inference(**kwargs)
