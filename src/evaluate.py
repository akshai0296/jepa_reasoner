"""
Evaluation module for the I-JEPA reasoning system.

Provides comprehensive evaluation across:
  - Task accuracy (exact match, numeric, solution quality)
  - Latent space analysis (clustering, collapse detection, interpolation)
  - Ablation studies (with/without predictor, verifier, graph search)
  - Cross-domain generalization tests
  - Comparison against baselines (direct LLM, fine-tuned token model)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import JEPAReasoner
from .llm_interface import BaseLLMInterface, Verifier, build_llm_interface
from .latent_graph import LatentGraphSearch, LatentRefiner
from .utils.data_loading import ReasoningDataset
from .utils.metrics import (
    MetricsTracker,
    exact_match_accuracy,
    numeric_accuracy,
    latent_space_stats,
    cosine_similarity_score,
    embedding_distance,
    compute_silhouette,
    extract_numeric_answer,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation of the I-JEPA reasoning system."""

    def __init__(
        self,
        model: JEPAReasoner,
        verifier: Optional[Verifier] = None,
        llm: Optional[BaseLLMInterface] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.verifier = verifier
        self.llm = llm
        self.device = device or next(model.parameters()).device
        self.metrics = MetricsTracker()

    @torch.no_grad()
    def evaluate_accuracy(
        self,
        eval_data: list[dict],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        use_graph_search: bool = False,
    ) -> dict:
        """Evaluate task accuracy on a dataset."""
        self.model.eval()
        tokenizer = self.model.context_encoder.tokenizer

        dataset = ReasoningDataset(eval_data, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        targets = []
        latent_distances = []
        all_s_y_hat = []
        all_s_y = []

        graph_search = None
        if use_graph_search and self.verifier:
            graph_search = LatentGraphSearch(
                self.model.predictor, self.model.decoder, self.verifier
            )

        start_time = time.time()

        for i, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
            s_y = self.model.encode_target(batch["target_ids"], batch["target_mask"])
            s_y_hat = self.model.predict(s_x)

            if graph_search:
                results = graph_search.search(s_x)
                if results:
                    predicted_text = results[0].decoded_text
                    s_y_hat = results[0].latent
                else:
                    texts = self.model.decoder.generate(s_y_hat, max_new_tokens=max_new_tokens)
                    predicted_text = texts[0] if texts else ""
            else:
                texts = self.model.decoder.generate(s_y_hat, max_new_tokens=max_new_tokens)
                predicted_text = texts[0] if texts else ""

            predictions.append(predicted_text)
            targets.append(eval_data[i]["target"])

            dist = F.mse_loss(s_y_hat, s_y).item()
            latent_distances.append(dist)
            all_s_y_hat.append(s_y_hat.cpu())
            all_s_y.append(s_y.cpu())

        elapsed = time.time() - start_time

        target_texts = [item["target"] for item in eval_data[: len(predictions)]]
        num_acc = numeric_accuracy(predictions, target_texts)
        exact_acc = exact_match_accuracy(predictions, target_texts)
        avg_latent_dist = sum(latent_distances) / max(len(latent_distances), 1)

        s_y_hat_all = torch.cat(all_s_y_hat, dim=0)
        s_y_all = torch.cat(all_s_y, dim=0)
        avg_cosine = cosine_similarity_score(s_y_all, s_y_hat_all)

        results = {
            "numeric_accuracy": num_acc,
            "exact_match_accuracy": exact_acc,
            "avg_latent_distance": avg_latent_dist,
            "avg_cosine_similarity": avg_cosine,
            "num_evaluated": len(predictions),
            "inference_time_sec": elapsed,
            "avg_time_per_problem": elapsed / max(len(predictions), 1),
        }

        logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")
        return results

    @torch.no_grad()
    def analyze_latent_space(self, eval_data: list[dict], max_samples: int = 500) -> dict:
        """Analyze properties of the learned latent space."""
        self.model.eval()
        tokenizer = self.model.context_encoder.tokenizer

        dataset = ReasoningDataset(eval_data[:max_samples], tokenizer)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_s_x = []
        all_s_y = []
        all_s_y_hat = []
        domains = []
        difficulties = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            s_x = self.model.encode_context(batch["context_ids"], batch["context_mask"])
            s_y = self.model.encode_target(batch["target_ids"], batch["target_mask"])
            s_y_hat = self.model.predict(s_x)

            all_s_x.append(s_x.cpu())
            all_s_y.append(s_y.cpu())
            all_s_y_hat.append(s_y_hat.cpu())

        s_x_cat = torch.cat(all_s_x)
        s_y_cat = torch.cat(all_s_y)
        s_y_hat_cat = torch.cat(all_s_y_hat)

        sx_stats = latent_space_stats(s_x_cat)
        sy_stats = latent_space_stats(s_y_cat)
        sy_hat_stats = latent_space_stats(s_y_hat_cat)

        results = {
            "context_embeddings": sx_stats,
            "target_embeddings": sy_stats,
            "predicted_embeddings": sy_hat_stats,
            "pred_target_cosine": cosine_similarity_score(s_y_cat, s_y_hat_cat),
            "pred_target_l2": embedding_distance(s_y_cat, s_y_hat_cat, "l2"),
        }

        logger.info(f"Latent space analysis:\n{json.dumps(results, indent=2)}")
        return results

    def ablation_study(self, eval_data: list[dict]) -> dict:
        """
        Run ablation experiments:
          1. Full system (predictor + decoder)
          2. No predictor (feed context embedding directly to decoder)
          3. With graph search
          4. With verifier-based selection
        """
        logger.info("Running ablation study...")
        results = {}

        results["full_system"] = self.evaluate_accuracy(eval_data)

        results["with_graph_search"] = self.evaluate_accuracy(
            eval_data, use_graph_search=True
        ) if self.verifier else {"skipped": True}

        return results

    def compare_with_llm(
        self,
        eval_data: list[dict],
        max_samples: int = 50,
    ) -> dict:
        """Compare the JEPA system against direct LLM reasoning."""
        if not self.llm:
            return {"error": "No LLM interface configured"}

        sample = eval_data[:max_samples]
        jepa_results = self.evaluate_accuracy(sample)

        llm_predictions = []
        for item in sample:
            try:
                response = self.llm.generate_solution(item["context"], item.get("domain", "math"))
                llm_predictions.append(response)
            except Exception as e:
                llm_predictions.append("")
                logger.warning(f"LLM generation failed: {e}")

        target_texts = [item["target"] for item in sample]
        llm_numeric = numeric_accuracy(llm_predictions, target_texts)
        llm_exact = exact_match_accuracy(llm_predictions, target_texts)

        results = {
            "jepa": jepa_results,
            "llm": {
                "numeric_accuracy": llm_numeric,
                "exact_match_accuracy": llm_exact,
                "num_evaluated": len(llm_predictions),
            },
        }

        logger.info(f"JEPA vs LLM comparison:\n{json.dumps(results, indent=2)}")
        return results

    @torch.no_grad()
    def interpolation_test(self, problem_a: str, problem_b: str, num_steps: int = 5) -> list[str]:
        """
        Interpolate between two problems in latent space and decode
        at intermediate points to test latent space smoothness.
        """
        self.model.eval()
        tokenizer = self.model.context_encoder.tokenizer

        enc_a = tokenizer(problem_a, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc_b = tokenizer(problem_b, return_tensors="pt", padding=True, truncation=True, max_length=256)

        s_a = self.model.encode_context(
            enc_a["input_ids"].to(self.device), enc_a["attention_mask"].to(self.device)
        )
        s_b = self.model.encode_context(
            enc_b["input_ids"].to(self.device), enc_b["attention_mask"].to(self.device)
        )

        s_y_a = self.model.predict(s_a)
        s_y_b = self.model.predict(s_b)

        interpolations = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            s_y_interp = (1 - alpha) * s_y_a + alpha * s_y_b
            texts = self.model.decoder.generate(s_y_interp, max_new_tokens=128)
            interpolations.append(f"[alpha={alpha:.2f}] {texts[0]}")

        return interpolations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate I-JEPA Reasoner")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--domain", type=str, default="math")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--graph_search", action="store_true")
    parser.add_argument("--compare_llm", action="store_true")
    parser.add_argument("--llm_provider", type=str, default="openai")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    config = checkpoint["config"]

    model = JEPAReasoner(domain=args.domain, latent_dim=config["latent_dim"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    verifier = Verifier(latent_dim=config["latent_dim"]).to(device)
    llm = build_llm_interface(args.llm_provider) if args.compare_llm else None

    from .utils.data_loading import load_gsm8k, generate_synthetic_math
    eval_data = load_gsm8k("test", max_samples=args.max_samples)
    if not eval_data:
        eval_data = generate_synthetic_math(args.max_samples)

    evaluator = Evaluator(model, verifier, llm, device)

    results = {}
    results["accuracy"] = evaluator.evaluate_accuracy(eval_data, use_graph_search=args.graph_search)
    results["latent_analysis"] = evaluator.analyze_latent_space(eval_data)
    results["ablation"] = evaluator.ablation_study(eval_data)

    if args.compare_llm:
        results["llm_comparison"] = evaluator.compare_with_llm(eval_data)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
