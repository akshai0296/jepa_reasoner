"""
LLM Inference Module for the I-JEPA reasoning system.

Provides an interface to external LLMs (OpenAI, Anthropic, local models)
for:
  1. Generating baseline/oracle solutions
  2. Providing feedback and critiques on the system's outputs
  3. Generating synthetic training data
  4. Scoring/verifying candidate solutions
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def score_solution(self, problem: str, solution: str) -> dict:
        ...


class OpenAIInterface(BaseLLMInterface):
    def __init__(self, model: str = "gpt-4", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            self.client = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful reasoning assistant.",
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self.client is None:
            raise RuntimeError("openai package not installed")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def score_solution(self, problem: str, solution: str) -> dict:
        prompt = f"""Evaluate the following solution to a problem. Score it on:
1. Correctness (0-10): Is the answer mathematically/logically correct?
2. Completeness (0-10): Are all steps shown?
3. Clarity (0-10): Is the reasoning clear?

Problem: {problem}

Solution: {solution}

Respond in JSON format: {{"correctness": X, "completeness": X, "clarity": X, "feedback": "..."}}"""

        response = self.generate(prompt, max_tokens=512)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            return {"correctness": 0, "completeness": 0, "clarity": 0, "feedback": response}

    def generate_solution(self, problem: str, domain: str = "math") -> str:
        domain_prompts = {
            "math": "Solve the following math problem step by step. Show your work clearly.",
            "code": "Write clean, working code to solve the following problem. Include comments.",
            "text": "Answer the following question with clear reasoning. Explain your thought process.",
        }
        system_prompt = domain_prompts.get(domain, domain_prompts["text"])
        return self.generate(problem, system_prompt=system_prompt)

    def critique_attempt(self, problem: str, attempt: str, correct_solution: str = "") -> str:
        prompt = f"""Analyze where this solution attempt went wrong and how to fix it.

Problem: {problem}

Attempt: {attempt}
"""
        if correct_solution:
            prompt += f"\nCorrect solution for reference: {correct_solution}"

        prompt += "\n\nProvide specific feedback on what went wrong and how to improve."
        return self.generate(prompt, max_tokens=512)


class AnthropicInterface(BaseLLMInterface):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            self.client = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful reasoning assistant.",
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self.client is None:
            raise RuntimeError("anthropic package not installed")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.content[0].text

    def score_solution(self, problem: str, solution: str) -> dict:
        prompt = f"""Evaluate the following solution. Score it on:
1. Correctness (0-10)
2. Completeness (0-10)
3. Clarity (0-10)

Problem: {problem}
Solution: {solution}

Respond in JSON: {{"correctness": X, "completeness": X, "clarity": X, "feedback": "..."}}"""

        response = self.generate(prompt, max_tokens=512)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            return {"correctness": 0, "completeness": 0, "clarity": 0, "feedback": response}


class LocalLLMInterface(BaseLLMInterface):
    """Interface for locally-hosted models via HuggingFace transformers."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map=self.device
        )

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
            )
        return self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    def score_solution(self, problem: str, solution: str) -> dict:
        prompt = f"Rate correctness 0-10.\nProblem: {problem}\nSolution: {solution}\nScore:"
        response = self.generate(prompt, max_tokens=64)
        try:
            score = int("".join(c for c in response[:10] if c.isdigit()) or "0")
            return {"correctness": min(score, 10), "feedback": response}
        except ValueError:
            return {"correctness": 0, "feedback": response}


class Verifier(torch.nn.Module):
    """
    Learned verifier that scores (problem_embedding, solution_embedding) pairs.
    Trained on correct/incorrect solutions to pick the best candidate.
    """

    def __init__(self, latent_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_x: torch.Tensor, s_y: torch.Tensor) -> torch.Tensor:
        """Returns a scalar score for each (s_x, s_y) pair."""
        combined = torch.cat([s_x, s_y], dim=-1)
        return self.net(combined).squeeze(-1)

    def rank_candidates(self, s_x: torch.Tensor, candidate_latents: list[torch.Tensor]) -> list[int]:
        """Rank candidate latents by verifier score (highest first)."""
        scores = []
        for s_y in candidate_latents:
            score = self(s_x, s_y)
            scores.append(score.mean().item())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


LLM_REGISTRY = {
    "openai": OpenAIInterface,
    "anthropic": AnthropicInterface,
    "local": LocalLLMInterface,
}


def build_llm_interface(provider: str = "openai", **kwargs) -> BaseLLMInterface:
    if provider not in LLM_REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return LLM_REGISTRY[provider](**kwargs)
