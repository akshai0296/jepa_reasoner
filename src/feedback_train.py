"""
Feedback-driven self-improvement loop for the I-JEPA reasoning system.

This module implements the continual self-improvement cycle:
  1. Evaluate the model on a set of problems
  2. Identify failures and low-confidence predictions
  3. Use an LLM to generate corrections and feedback
  4. Create new training examples from the feedback
  5. Fine-tune the model on the augmented data
  6. Repeat

Also trains the Verifier network on correct/incorrect solution pairs.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import JEPAReasoner
from .llm_interface import BaseLLMInterface, Verifier, build_llm_interface
from .latent_graph import LatentGraphSearch, LatentRefiner, cluster_latents
from .utils.data_loading import ReasoningDataset
from .utils.metrics import MetricsTracker, numeric_accuracy, extract_numeric_answer

logger = logging.getLogger(__name__)


class FeedbackTrainer:
    """
    Implements the self-improvement feedback loop.
    """

    def __init__(
        self,
        model: JEPAReasoner,
        verifier: Verifier,
        llm: BaseLLMInterface,
        train_data: list[dict],
        config: dict,
    ):
        self.model = model
        self.verifier = verifier
        self.llm = llm
        self.train_data = train_data
        self.config = config
        self.device = next(model.parameters()).device
        self.metrics = MetricsTracker()
        self.feedback_data = []

        self.verifier.to(self.device)

    def run_improvement_cycle(
        self,
        eval_data: list[dict],
        num_cycles: int = 5,
        problems_per_cycle: int = 50,
    ):
        """
        Execute multiple self-improvement cycles.

        Each cycle: evaluate -> identify failures -> get feedback -> retrain
        """
        logger.info(f"Starting {num_cycles} self-improvement cycles")

        for cycle in range(num_cycles):
            logger.info(f"\n=== Improvement Cycle {cycle + 1}/{num_cycles} ===")

            sample = random.sample(eval_data, min(problems_per_cycle, len(eval_data)))
            failures, successes = self._evaluate_batch(sample)

            logger.info(
                f"Cycle {cycle+1}: {len(successes)} successes, {len(failures)} failures "
                f"({len(successes)/(len(successes)+len(failures))*100:.1f}% accuracy)"
            )

            if failures:
                new_examples = self._generate_feedback(failures)
                self.feedback_data.extend(new_examples)
                logger.info(f"Generated {len(new_examples)} feedback examples")

            self._train_verifier(successes, failures)

            if self.feedback_data:
                self._finetune_on_feedback()

            self.metrics.update(
                {
                    "cycle": cycle + 1,
                    "accuracy": len(successes) / max(len(successes) + len(failures), 1),
                    "feedback_pool_size": len(self.feedback_data),
                },
                step=cycle,
            )

        return self.metrics.summary()

    @torch.no_grad()
    def _evaluate_batch(self, data: list[dict]) -> tuple[list[dict], list[dict]]:
        """Evaluate the model on a batch of problems, return (failures, successes)."""
        self.model.eval()
        tokenizer = self.model.context_encoder.tokenizer

        failures = []
        successes = []

        for item in data:
            enc = tokenizer(
                item["context"],
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            context_ids = enc["input_ids"].to(self.device)
            context_mask = enc["attention_mask"].to(self.device)

            result = self.model.inference(
                context_ids, context_mask, max_new_tokens=128, num_candidates=1
            )

            predicted_text = result["candidates"][0][0] if result["candidates"] else ""

            is_correct = self._check_correctness(predicted_text, item)

            record = {
                **item,
                "predicted": predicted_text,
                "s_x": result["s_x"],
                "s_y_hat": result["latents"][0],
                "correct": is_correct,
            }

            if is_correct:
                successes.append(record)
            else:
                failures.append(record)

        return failures, successes

    def _check_correctness(self, predicted: str, item: dict) -> bool:
        """Check if a prediction is correct."""
        if "final_answer" in item:
            pred_num = extract_numeric_answer(predicted)
            try:
                target_num = float(item["final_answer"])
                if pred_num is not None:
                    return abs(pred_num - target_num) < 1e-3
            except (ValueError, TypeError):
                pass

        target = item["target"].strip().lower()
        pred = predicted.strip().lower()

        if target in pred or pred in target:
            return True

        return False

    def _generate_feedback(self, failures: list[dict], max_examples: int = 20) -> list[dict]:
        """Use the LLM to generate corrective feedback for failed problems."""
        new_examples = []
        sample = failures[:max_examples]

        for item in sample:
            try:
                correct_solution = self.llm.generate_solution(
                    item["context"], domain=item.get("domain", "math")
                )

                new_examples.append(
                    {
                        "context": item["context"],
                        "target": correct_solution,
                        "domain": item.get("domain", "math"),
                        "difficulty": item.get("difficulty", 1),
                        "source": "llm_feedback",
                    }
                )

                critique = self.llm.critique_attempt(
                    item["context"],
                    item["predicted"],
                    correct_solution,
                )

                new_examples.append(
                    {
                        "context": f"{item['context']}\n\nPrevious attempt (incorrect): {item['predicted']}\n\nFeedback: {critique}",
                        "target": correct_solution,
                        "domain": item.get("domain", "math"),
                        "difficulty": item.get("difficulty", 1),
                        "source": "error_correction",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to generate feedback: {e}")
                continue

        return new_examples

    def _train_verifier(
        self,
        successes: list[dict],
        failures: list[dict],
        num_epochs: int = 5,
    ):
        """Train the verifier on correct/incorrect solution pairs."""
        if not successes or not failures:
            return

        self.verifier.train()
        optimizer = torch.optim.AdamW(self.verifier.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            epoch_loss = 0
            count = 0

            for correct in successes:
                if "s_x" not in correct or "s_y_hat" not in correct:
                    continue
                s_x = correct["s_x"]
                s_y_pos = correct["s_y_hat"]

                neg = random.choice(failures)
                if "s_y_hat" not in neg:
                    continue
                s_y_neg = neg["s_y_hat"]

                pos_score = self.verifier(s_x, s_y_pos)
                neg_score = self.verifier(s_x, s_y_neg)

                loss = F.relu(1.0 - pos_score + neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += 1

            if count > 0:
                logger.info(f"Verifier epoch {epoch+1}: loss={epoch_loss/count:.4f}")

    def _finetune_on_feedback(self, num_epochs: int = 3):
        """Fine-tune the model on accumulated feedback data."""
        logger.info(f"Fine-tuning on {len(self.feedback_data)} feedback examples")

        tokenizer = self.model.context_encoder.tokenizer
        dataset = ReasoningDataset(self.feedback_data, tokenizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        params = list(self.model.predictor.parameters()) + list(self.model.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            count = 0

            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = self.model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    target_ids=batch["target_ids"],
                    target_mask=batch["target_mask"],
                    solution_ids=batch["solution_ids"],
                    solution_mask=batch["solution_mask"],
                )
                loss = outputs["total_loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                count += 1

            logger.info(f"Feedback finetune epoch {epoch+1}: loss={epoch_loss/count:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run self-improvement feedback loop")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--llm_provider", type=str, default="openai")
    parser.add_argument("--num_cycles", type=int, default=5)
    parser.add_argument("--domain", type=str, default="math")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]

    model = JEPAReasoner(domain=args.domain, latent_dim=config["latent_dim"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    verifier = Verifier(latent_dim=config["latent_dim"])
    llm = build_llm_interface(args.llm_provider)

    from .utils.data_loading import load_gsm8k, generate_synthetic_math

    eval_data = load_gsm8k("test", max_samples=200)
    if not eval_data:
        eval_data = generate_synthetic_math(200)

    trainer = FeedbackTrainer(model, verifier, llm, [], config)
    results = trainer.run_improvement_cycle(eval_data, num_cycles=args.num_cycles)
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()
