#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="${1:-checkpoints/final.pt}"
LLM_PROVIDER="${2:-openai}"
NUM_CYCLES="${3:-5}"

echo "Running self-improvement loop"
echo "Checkpoint: $CHECKPOINT"
echo "LLM Provider: $LLM_PROVIDER"
echo "Cycles: $NUM_CYCLES"

python -m src.feedback_train \
    --checkpoint "$CHECKPOINT" \
    --llm_provider "$LLM_PROVIDER" \
    --num_cycles "$NUM_CYCLES" \
    --domain math
