"""
Self-supervised training pipeline for the I-JEPA reasoning system.

Training proceeds in stages:
  Stage 1: Train encoders + predictor with latent prediction + contrastive loss
  Stage 2: Train decoder conditioned on predicted latents (with optional curriculum)
  Stage 3: Joint fine-tuning of the full pipeline

Supports curriculum learning, mixed-precision training, and gradient checkpointing.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from .models import JEPAReasoner, MultiDomainJEPAReasoner
from .utils.data_loading import (
    ReasoningDataset,
    CurriculumSampler,
    build_dataloader,
    generate_synthetic_math,
    load_gsm8k,
    load_math_dataset,
    load_code_dataset,
    load_text_reasoning,
)
from .utils.metrics import MetricsTracker, latent_space_stats, embedding_distance

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the multi-stage training of JEPAReasoner.
    """

    def __init__(
        self,
        model: JEPAReasoner,
        train_data: list[dict],
        val_data: list[dict],
        config: dict,
    ):
        self.model = model
        self.config = config
        device_cfg = config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device_cfg)
        self.model.to(self.device)

        self.tokenizer = model.context_encoder.tokenizer
        self.metrics = MetricsTracker()

        self.train_dataset = ReasoningDataset(
            train_data,
            self.tokenizer,
            max_context_len=config.get("max_context_len", 256),
            max_target_len=config.get("max_target_len", 256),
        )
        self.val_dataset = ReasoningDataset(
            val_data,
            self.tokenizer,
            max_context_len=config.get("max_context_len", 256),
            max_target_len=config.get("max_target_len", 256),
        )

        self.curriculum = CurriculumSampler(
            self.train_dataset, num_epochs=config.get("num_epochs", 50)
        )

        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.output_dir = Path(config.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, stage: str):
        lr = self.config.get(f"lr_{stage}", self.config.get("lr", 1e-4))
        wd = self.config.get("weight_decay", 0.01)

        if stage == "predictor":
            params = list(self.model.predictor.parameters())
            if not self.config.get("freeze_backbone", False):
                params += list(self.model.context_encoder.parameters())
        elif stage == "decoder":
            params = list(self.model.decoder.parameters())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self, optimizer, num_steps):
        warmup_steps = min(num_steps // 10, 500)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=optimizer.defaults["lr"], total_steps=num_steps,
            pct_start=warmup_steps / max(num_steps, 1),
        )

    def train_predictor(self, num_epochs: Optional[int] = None):
        """Stage 1: Train encoders + predictor on latent prediction loss."""
        num_epochs = num_epochs or self.config.get("predictor_epochs", 30)
        batch_size = self.config.get("batch_size", 8)
        grad_clip = self.config.get("grad_clip", 1.0)
        loss_type = self.config.get("loss_type", "l2")
        contrastive_weight = self.config.get("contrastive_weight", 0.1)

        optimizer = self._build_optimizer("predictor")
        best_val_loss = float("inf")
        global_step = 0

        logger.info("=== Stage 1: Training Predictor ===")

        for epoch in range(num_epochs):
            self.model.train()
            self.model.decoder.eval()

            indices = self.curriculum.get_indices(epoch)
            subset = Subset(self.train_dataset, indices)
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )

            epoch_losses = []
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
                    s_y = self.model.encode_target(batch["target_ids"], batch["target_mask"])
                    s_y_hat = self.model.predict(s_x)

                    pred_loss = self.model.prediction_loss(s_y, s_y_hat, loss_type)
                    c_loss = self.model.contrastive_loss(s_y_hat, s_y)
                    loss = pred_loss + contrastive_weight * c_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

                self.model.target_encoder.ema_update(self.model.context_encoder)

                epoch_losses.append(loss.item())
                self.metrics.update(
                    {"pred_loss": pred_loss.item(), "contrastive_loss": c_loss.item()},
                    global_step,
                )
                global_step += 1

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            val_loss = self._validate_predictor(loss_type)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("predictor_best")

            if (epoch + 1) % 10 == 0:
                self._check_latent_health()

        return best_val_loss

    @torch.no_grad()
    def _validate_predictor(self, loss_type: str = "l2") -> float:
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        total_loss = 0
        count = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
            s_y = self.model.encode_target(batch["target_ids"], batch["target_mask"])
            s_y_hat = self.model.predict(s_x)

            loss = self.model.prediction_loss(s_y, s_y_hat, loss_type)
            total_loss += loss.item() * s_x.size(0)
            count += s_x.size(0)

        return total_loss / max(count, 1)

    def train_decoder(self, num_epochs: Optional[int] = None, use_true_latent_ratio: float = 0.5):
        """
        Stage 2: Train decoder conditioned on predicted latents.

        Initially uses true s_y (from target encoder) for a fraction of
        examples to help the decoder learn, then transitions to using
        predicted s_y_hat as training progresses.
        """
        num_epochs = num_epochs or self.config.get("decoder_epochs", 20)
        batch_size = self.config.get("batch_size", 8)
        grad_clip = self.config.get("grad_clip", 1.0)

        optimizer = self._build_optimizer("decoder")
        loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        logger.info("=== Stage 2: Training Decoder ===")

        for epoch in range(num_epochs):
            self.model.train()
            for p in self.model.context_encoder.parameters():
                p.requires_grad = False
            for p in self.model.predictor.parameters():
                p.requires_grad = False

            true_ratio = max(0.0, use_true_latent_ratio * (1 - epoch / num_epochs))
            epoch_losses = []

            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.no_grad():
                    s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
                    s_y = self.model.encode_target(batch["target_ids"], batch["target_mask"])
                    s_y_hat = self.model.predict(s_x)

                use_true = torch.rand(1).item() < true_ratio
                latent_input = s_y if use_true else s_y_hat

                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    dec_out = self.model.decode(
                        latent_input,
                        batch["solution_ids"],
                        batch["solution_mask"],
                    )
                    loss = dec_out["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(
                f"Decoder Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | True ratio: {true_ratio:.2f}"
            )

        self._save_checkpoint("decoder_trained")

        for p in self.model.context_encoder.parameters():
            p.requires_grad = True
        for p in self.model.predictor.parameters():
            p.requires_grad = True

    def joint_finetune(self, num_epochs: Optional[int] = None):
        """Stage 3: Joint fine-tuning of the entire pipeline."""
        num_epochs = num_epochs or self.config.get("finetune_epochs", 10)
        batch_size = self.config.get("batch_size", 8)
        grad_clip = self.config.get("grad_clip", 1.0)

        optimizer = self._build_optimizer("joint")
        steps_per_epoch = (len(self.train_dataset) + batch_size - 1) // batch_size
        scheduler = self._build_scheduler(
            optimizer, num_epochs * steps_per_epoch
        )

        logger.info("=== Stage 3: Joint Fine-tuning ===")

        for epoch in range(num_epochs):
            self.model.train()
            loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            epoch_losses = []

            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        context_ids=batch["context_ids"],
                        context_mask=batch["context_mask"],
                        target_ids=batch["target_ids"],
                        target_mask=batch["target_mask"],
                        solution_ids=batch["solution_ids"],
                        solution_mask=batch["solution_mask"],
                        loss_type=self.config.get("loss_type", "l2"),
                    )
                    loss = outputs["total_loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Joint Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        self._save_checkpoint("joint_finetuned")

    @torch.no_grad()
    def _check_latent_health(self):
        """Monitor latent space for collapse or degeneration."""
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        all_s_y_hat = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
            s_y_hat = self.model.predict(s_x)
            all_s_y_hat.append(s_y_hat)
            if len(all_s_y_hat) * 32 >= 200:
                break

        embeddings = torch.cat(all_s_y_hat, dim=0)[:200]
        stats = latent_space_stats(embeddings)

        if stats["collapse_detected"]:
            logger.warning(
                f"LATENT COLLAPSE DETECTED! Avg pairwise cosine: {stats['avg_pairwise_cosine']:.4f}"
            )
        else:
            logger.info(
                f"Latent health: variance={stats['variance']:.4f}, "
                f"avg_cos={stats['avg_pairwise_cosine']:.4f}"
            )

    def _save_checkpoint(self, name: str):
        path = self.output_dir / f"{name}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "metrics": self.metrics.to_dict(),
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded checkpoint: {path}")

    def run_full_pipeline(self):
        """Execute the complete multi-stage training pipeline."""
        logger.info("Starting full training pipeline...")
        start = time.time()

        self.train_predictor()
        self.train_decoder()
        self.joint_finetune()

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed/60:.1f} minutes")
        self._save_checkpoint("final")

        return self.metrics.summary()


def load_training_data(config: dict) -> tuple[list[dict], list[dict]]:
    """Load and split training data based on config."""
    all_data = []
    max_samples = config.get("max_samples_per_dataset", 5000)

    domains = config.get("domains", ["math"])

    for domain in domains:
        if domain == "math":
            data = load_gsm8k("train", max_samples)
            if not data:
                logger.info("Falling back to synthetic math data")
                data = generate_synthetic_math(max_samples)
            all_data.extend(data)

        elif domain == "code":
            data = load_code_dataset("train", max_samples)
            all_data.extend(data)

        elif domain == "text":
            data = load_text_reasoning("train", max_samples)
            all_data.extend(data)

    if not all_data:
        logger.warning("No data loaded, generating synthetic data")
        all_data = generate_synthetic_math(1000)

    import random
    random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    return all_data[:split], all_data[split:]


def main():
    parser = argparse.ArgumentParser(description="Train I-JEPA Reasoner")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--domain", type=str, default="math", choices=["math", "code", "text"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "domains": [args.domain],
            "latent_dim": args.latent_dim,
            "predictor_epochs": args.epochs,
            "decoder_epochs": args.epochs // 2,
            "finetune_epochs": args.epochs // 3,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_amp": args.use_amp,
            "output_dir": args.output_dir,
            "max_samples_per_dataset": args.max_samples,
            "loss_type": "l2",
            "contrastive_weight": 0.1,
            "freeze_backbone": False,
            "grad_clip": 1.0,
            "weight_decay": 0.01,
        }

    train_data, val_data = load_training_data(config)
    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val examples")

    predictor_kwargs = config.get("predictor_kwargs", {"hidden_dim": 1024, "num_layers": 4, "num_heads": 8})
    decoder_kwargs = config.get("decoder_kwargs", {"hidden_dim": 512, "num_layers": 4, "num_heads": 8})
    decoder_kwargs.setdefault("latent_dim", config["latent_dim"])

    model = JEPAReasoner(
        domain=config["domains"][0],
        latent_dim=config["latent_dim"],
        predictor_type=config.get("predictor_type", "transformer"),
        predictor_kwargs=predictor_kwargs,
        decoder_type=config.get("decoder_type", "scratch"),
        decoder_kwargs=decoder_kwargs,
        freeze_backbone=config.get("freeze_backbone", False),
    )

    trainer = Trainer(model, train_data, val_data, config)
    results = trainer.run_full_pipeline()
    logger.info(f"Final metrics: {results}")


if __name__ == "__main__":
    main()
