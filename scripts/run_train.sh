#!/usr/bin/env bash
# Reproduce the paper's main experiment (BERT-Large, 5 epochs, BCL).
set -euo pipefail
ROOT="${ROOT:-data/WSD_Evaluation_Framework}"

python -m src.train \
    --train_file "$ROOT/Training_Corpora/SemCor/semcor.data.xml" \
    --train_key  "$ROOT/Training_Corpora/SemCor/semcor.gold.key.txt" \
    --dev_file   "$ROOT/Evaluation_Datasets/semeval2007/semeval2007.data.xml" \
    --dev_key    "$ROOT/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt" \
    --output_dir runs/polybert-large \
    --bert_model bert-large-uncased \
    --poly_m 16 --num_heads 8 \
    --batch_size 32 --epochs 5 --lr 1e-5 \
    --max_seq_len 160 --max_gloss_len 32 \
    --fp16
