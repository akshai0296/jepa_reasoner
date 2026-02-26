# I-JEPA Reasoner: Cross-Domain Latent Reasoning System

A self-improving reasoning system that **thinks in embeddings and talks in language**. Inspired by Yann LeCun's Joint Embedding Predictive Architecture (JEPA), this system predicts abstract latent representations of solutions instead of generating tokens directly — then decodes those latents into human-readable code, math solutions, or natural language explanations.

## Core Idea

Traditional LLMs reason token-by-token, committing to a single path and accumulating errors. This system takes a fundamentally different approach:

1. **Encode** the problem into a latent vector
2. **Predict** the solution's semantic representation in embedding space (not tokens)
3. **Decode** the latent prediction into human-readable output
4. **Verify & refine** using a learned verifier and latent graph search

By separating *reasoning* (latent prediction) from *speaking* (token generation), the system can explore multiple solution paths simultaneously in latent space, avoid premature commitment, and self-improve through feedback loops.

## Architecture

```
Problem x ──→ [Context Encoder] ──→ s_x ──→ [Predictor(s_x, z)] ──→ ŝ_y ──→ [Decoder] ──→ ŷ (text)
                                                                       ↑
Solution y ──→ [Target Encoder] ──→ s_y ─── training loss: D(s_y, ŝ_y)
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Context Encoder** | Encodes problem text into latent vector s_x | Domain-specific: CodeBERT (code), MiniLM (math/text) |
| **Target Encoder** | Encodes solution into s_y (EMA of context encoder) | Same architecture, updated via exponential moving average |
| **Latent Predictor** | Predicts s_y from s_x — the "world model" | Transformer (4-6 layers) or MLP |
| **Decoder (Talker)** | Converts predicted ŝ_y into text tokens | GPT-2-style with latent prefix conditioning |
| **Verifier** | Scores (problem, solution) pairs for candidate selection | MLP binary classifier on concatenated embeddings |
| **LLM Interface** | External LLM for feedback, critiques, synthetic data | OpenAI, Anthropic, or local LLaMA |

### Supported Domains

- **Math**: GSM8K, MATH competition, DeepMind Mathematics, synthetic arithmetic
- **Code**: CodeSearchNet, function docstring → implementation
- **Text**: HotpotQA, logical reasoning, commonsense QA

## Quick Start

### Installation

```bash
cd jepa_reasoner
pip install -r requirements.txt
```

### Train on Math (default)

```bash
# Using synthetic data (no downloads needed)
python -m src.train --domain math --epochs 30 --batch_size 8

# Using a config file
python -m src.train --config configs/default.json

# Smaller/faster config for testing
python -m src.train --config configs/math_small.json
```

### Download Datasets

```bash
bash scripts/download_data.sh
```

### Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/final.pt --domain math
```

### Self-Improvement Loop

```bash
# Requires OPENAI_API_KEY or ANTHROPIC_API_KEY
python -m src.feedback_train \
    --checkpoint checkpoints/final.pt \
    --llm_provider openai \
    --num_cycles 5
```

## Training Pipeline

Training proceeds in three stages:

### Stage 1: Predictor Training
Train the context encoder + predictor to minimize latent prediction loss `D(s_y, ŝ_y)` plus an InfoNCE contrastive loss to prevent embedding collapse. Uses curriculum learning (easy → hard problems).

### Stage 2: Decoder Training
Freeze the predictor, train the decoder to generate solution tokens conditioned on the predicted latent. Starts with true s_y and gradually transitions to predicted ŝ_y.

### Stage 3: Joint Fine-tuning
End-to-end fine-tuning of the full pipeline with combined losses (prediction + contrastive + decoder cross-entropy).

## Latent Space Reasoning

Beyond single-pass prediction, the system supports advanced latent operations:

- **Latent Graph Search**: Beam search over latent states — generate multiple candidate latents via perturbation, decode and score each, expand the best
- **Iterative Denoising**: Apply the predictor repeatedly to refine noisy latents
- **Gradient-based Refinement**: Optimize the latent vector to maximize verifier score
- **Solution Clustering**: Group multiple solution attempts in latent space, pick the best cluster

## Self-Improvement Feedback Loop

The system continuously evaluates itself and improves:

1. Run evaluation on held-out problems
2. Identify failures (wrong answers, low verifier scores)
3. Query an LLM (GPT-4/Claude) for correct solutions and critiques
4. Generate new training examples from the feedback
5. Fine-tune the predictor and decoder on augmented data
6. Train the verifier on correct/incorrect pairs
7. Repeat

## Configuration

See `configs/` for example configurations:

| Config | Description |
|--------|-------------|
| `default.json` | Standard math training, transformer predictor |
| `math_small.json` | Lightweight config for testing, MLP predictor, frozen backbone |
| `multi_domain.json` | All three domains, larger predictor with latent z variable |

Key parameters:
- `latent_dim`: Embedding dimension (384 or 768)
- `predictor_type`: `"transformer"` or `"mlp"`
- `use_latent_z`: Enable stochastic latent variable for multi-modal predictions
- `loss_type`: `"l2"`, `"cosine"`, or `"smooth_l1"`
- `contrastive_weight`: Weight for InfoNCE loss (prevents collapse)

## Project Structure

```
jepa_reasoner/
├── src/
│   ├── encoders.py          # Context & target encoders (CodeBERT, MiniLM)
│   ├── predictors.py        # Latent predictor (Transformer / MLP)
│   ├── decoders.py          # Output decoders (latent-conditioned GPT-2)
│   ├── models.py            # JEPAReasoner: ties everything together
│   ├── llm_interface.py     # LLM integration + learned Verifier
│   ├── latent_graph.py      # Graph search, denoising, refinement
│   ├── train.py             # Multi-stage training pipeline
│   ├── feedback_train.py    # Self-improvement feedback loop
│   ├── evaluate.py          # Evaluation & ablation studies
│   └── utils/
│       ├── data_loading.py  # Dataset loading (GSM8K, MATH, CodeSearchNet, etc.)
│       └── metrics.py       # Metrics & latent space analysis
├── configs/                 # Training configurations
├── scripts/                 # Shell scripts for training/eval
├── data/                    # Dataset storage (code/, math/, text/)
├── models/                  # Saved model checkpoints
├── notebooks/               # Exploration & demo notebooks
└── docs/                    # Design documentation
```

## References

- LeCun et al. — Joint Embedding Predictive Architecture (JEPA)
- Assran et al. — I-JEPA: self-supervised learning from images
- Hao et al. — CoCoNut: Chain-of-Continuous-Thought for latent reasoning
- JEPA-Reasoner (ICLR 2026) — latent space reasoning with separate decoder
- Cobbe et al. — Training verifiers to solve math word problems
