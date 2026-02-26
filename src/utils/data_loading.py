"""
Dataset classes for loading math, code, and text reasoning data.

Supports loading from:
  - HuggingFace Datasets (GSM8K, MATH, CodeSearchNet, etc.)
  - Local JSON/JSONL files
  - Synthetic generation
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class ReasoningDataset(Dataset):
    """
    Generic dataset for (context, target) reasoning pairs.

    Each item is a dict with:
      - context: str (problem/question)
      - target: str  (solution/answer)
      - domain: str  (code/math/text)
      - difficulty: int (0=easy, 1=medium, 2=hard) for curriculum
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_context_len: int = 256,
        max_target_len: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        target = item["target"]

        context_enc = self.tokenizer(
            context,
            max_length=self.max_context_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "context_ids": context_enc["input_ids"].squeeze(0),
            "context_mask": context_enc["attention_mask"].squeeze(0),
            "target_ids": target_enc["input_ids"].squeeze(0),
            "target_mask": target_enc["attention_mask"].squeeze(0),
            "solution_ids": target_enc["input_ids"].squeeze(0),
            "solution_mask": target_enc["attention_mask"].squeeze(0),
            "domain": item.get("domain", "text"),
            "difficulty": item.get("difficulty", 0),
        }


class CurriculumSampler:
    """
    Samples data with curriculum learning: starts with easy examples
    and gradually introduces harder ones.
    """

    def __init__(self, dataset: ReasoningDataset, num_epochs: int = 100):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.current_epoch = 0

        self.by_difficulty = {}
        for idx, item in enumerate(dataset.data):
            diff = item.get("difficulty", 0)
            self.by_difficulty.setdefault(diff, []).append(idx)

    def get_indices(self, epoch: int) -> list[int]:
        """Get training indices for a given epoch, respecting curriculum."""
        progress = min(epoch / max(self.num_epochs * 0.5, 1), 1.0)

        indices = list(self.by_difficulty.get(0, []))

        if progress > 0.3:
            medium = self.by_difficulty.get(1, [])
            frac = min((progress - 0.3) / 0.3, 1.0)
            n = int(len(medium) * frac)
            indices.extend(random.sample(medium, min(n, len(medium))))

        if progress > 0.6:
            hard = self.by_difficulty.get(2, [])
            frac = min((progress - 0.6) / 0.4, 1.0)
            n = int(len(hard) * frac)
            indices.extend(random.sample(hard, min(n, len(hard))))

        random.shuffle(indices)
        return indices


def load_gsm8k(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """Load GSM8K dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
        data = []
        for item in ds:
            answer = item["answer"]
            final_answer = ""
            if "####" in answer:
                final_answer = answer.split("####")[-1].strip()

            data.append(
                {
                    "context": item["question"],
                    "target": answer,
                    "domain": "math",
                    "difficulty": 0 if len(answer.split("\n")) <= 4 else 1,
                    "final_answer": final_answer,
                }
            )
            if max_samples and len(data) >= max_samples:
                break
        return data
    except Exception as e:
        print(f"Could not load GSM8K: {e}")
        return []


def load_math_dataset(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """Load the MATH competition dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset("hendrycks/competition_math", split=split)
        data = []
        difficulty_map = {"Level 1": 0, "Level 2": 0, "Level 3": 1, "Level 4": 1, "Level 5": 2}
        for item in ds:
            data.append(
                {
                    "context": item["problem"],
                    "target": item["solution"],
                    "domain": "math",
                    "difficulty": difficulty_map.get(item.get("level", ""), 1),
                    "subject": item.get("type", ""),
                }
            )
            if max_samples and len(data) >= max_samples:
                break
        return data
    except Exception as e:
        print(f"Could not load MATH dataset: {e}")
        return []


def load_code_dataset(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """Load code reasoning data from CodeSearchNet (Python subset)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("code_search_net", "python", split=split, trust_remote_code=True)
        data = []
        for item in ds:
            docstring = item.get("func_documentation_string", "")
            code = item.get("func_code_string", "")
            if not docstring or not code or len(code) < 20:
                continue

            data.append(
                {
                    "context": docstring,
                    "target": code,
                    "domain": "code",
                    "difficulty": 0 if len(code.split("\n")) <= 10 else (1 if len(code.split("\n")) <= 30 else 2),
                }
            )
            if max_samples and len(data) >= max_samples:
                break
        return data
    except Exception as e:
        print(f"Could not load CodeSearchNet: {e}")
        return []


def load_text_reasoning(split: str = "train", max_samples: Optional[int] = None) -> list[dict]:
    """Load text reasoning data from HotpotQA or similar."""
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
        data = []
        for item in ds:
            context_sentences = []
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                context_sentences.extend(sentences)

            context_text = " ".join(context_sentences[:10])
            question = item["question"]
            answer = item["answer"]

            data.append(
                {
                    "context": f"Context: {context_text}\n\nQuestion: {question}",
                    "target": answer,
                    "domain": "text",
                    "difficulty": 0 if item.get("level", "") == "easy" else 1,
                }
            )
            if max_samples and len(data) >= max_samples:
                break
        return data
    except Exception as e:
        print(f"Could not load HotpotQA: {e}")
        return []


def load_local_data(path: str) -> list[dict]:
    """Load data from a local JSON or JSONL file."""
    data = []
    path = Path(path)

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif path.suffix == ".json":
        with open(path) as f:
            loaded = json.load(f)
            data = loaded if isinstance(loaded, list) else [loaded]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    for item in data:
        item.setdefault("domain", "text")
        item.setdefault("difficulty", 0)

    return data


def build_dataloader(
    data: list[dict],
    tokenizer,
    batch_size: int = 8,
    max_context_len: int = 256,
    max_target_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ReasoningDataset(data, tokenizer, max_context_len, max_target_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def generate_synthetic_math(num_samples: int = 1000) -> list[dict]:
    """Generate simple synthetic arithmetic problems for initial testing."""
    data = []
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}

    for _ in range(num_samples):
        op_name = random.choice(list(ops.keys()))
        if op_name == "*":
            a, b = random.randint(1, 20), random.randint(1, 20)
        else:
            a, b = random.randint(1, 100), random.randint(1, 100)

        result = ops[op_name](a, b)
        problem = f"What is {a} {op_name} {b}?"
        solution = f"{a} {op_name} {b} = {result}"

        data.append(
            {
                "context": problem,
                "target": solution,
                "domain": "math",
                "difficulty": 0,
                "final_answer": str(result),
            }
        )

    for _ in range(num_samples // 2):
        a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)
        op1, op2 = random.choice(["+", "-"]), random.choice(["+", "-"])
        expr = f"{a} {op1} {b} {op2} {c}"
        result = eval(expr)
        data.append(
            {
                "context": f"Calculate: {expr}",
                "target": f"{expr} = {result}",
                "domain": "math",
                "difficulty": 1,
                "final_answer": str(result),
            }
        )

    return data
