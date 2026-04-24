#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-data/WSD_Evaluation_Framework}"

python -m src.evaluate \
    --ckpt runs/polybert-large/best.pt \
    --eval_root "$ROOT/Evaluation_Datasets" \
    --bert_model bert-large-uncased
