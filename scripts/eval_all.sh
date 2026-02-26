#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="${1:-checkpoints/final.pt}"
DOMAIN="${2:-math}"
OUTPUT="${3:-eval_results.json}"

echo "Evaluating checkpoint: $CHECKPOINT"
echo "Domain: $DOMAIN"

python -m src.evaluate \
    --checkpoint "$CHECKPOINT" \
    --domain "$DOMAIN" \
    --max_samples 200 \
    --output "$OUTPUT"
