#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Downloading datasets via HuggingFace..."

python -c "
from datasets import load_dataset

print('Downloading GSM8K...')
load_dataset('openai/gsm8k', 'main')

print('Downloading MATH competition dataset...')
try:
    load_dataset('hendrycks/competition_math')
except:
    print('  (MATH dataset may require agreement - skipping)')

print('All available datasets downloaded.')
"
