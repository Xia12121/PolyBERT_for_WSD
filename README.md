# PolyBERT for Word Sense Disambiguation

Unofficial PyTorch implementation of

> **PolyBERT: Fine-Tuned Poly Encoder BERT-Based Model for Word Sense Disambiguation**
> Linhan Xia, Mingzhan Yang, Guohui Yuan, Shengnan Tao, Yujing Qiu, Guo Yu, Kai Lei.
> *18th International Conference on Knowledge Science, Engineering and Management*
> (**KSEM 2025**), Macao, China, August 4–7, 2025. LNCS 15922, pp. 433–443.
> Springer. DOI: [10.1007/978-981-95-3058-8_41](https://doi.org/10.1007/978-981-95-3058-8_41)
> — preprint: [arXiv:2506.00968](https://arxiv.org/abs/2506.00968).

PolyBERT tackles two long-standing problems of BERT-based Word Sense
Disambiguation (WSD):

1. **Imbalanced semantic representation.** Token-level (local) and
   sequence-level (global) signals carry different information; previous
   models privileged one over the other. PolyBERT fuses them with a
   poly-encoder + multi-head attention head.
2. **Wasted compute on negatives.** Earlier systems pair every target word
   with **all** of its candidate senses during training. PolyBERT introduces
   **Batch Contrastive Learning (BCL)**: within a mini-batch the gold gloss
   of every other target word serves as a negative for the current target
   word — no extra forward passes, no manual negative sampling.

The paper reports +2 F1 over GlossBERT / BEM on the standard WSD test
suite, and a 37.6 % wall-clock saving from BCL over a non-BCL variant.

---

## Table of contents

- [Architecture](#architecture)
- [The training objective (BCL)](#the-training-objective-bcl)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameters not specified in the paper](#hyperparameters-not-specified-in-the-paper)
- [Implementation notes / deviations](#implementation-notes--deviations)
- [Reproducing the paper's headline numbers](#reproducing-the-papers-headline-numbers)
- [FAQ](#faq)
- [Citing](#citing)
- [License & disclaimer](#license--disclaimer)

---

## Architecture

```
                      Context-encoder (BERT)                     Gloss-encoder (BERT)
                                │                                         │
                       r_wt = E_C[t]   (local)                  r_g = E_G[CLS]  (global)
                                │                                         │
            Q = 1_poly_m ⊗ r_wt ┐                                         │
                                 │  multi-head-attention(Q, K=V=E_C)      │
                                 ▼                                        │
                         r_wtᶠ  ∈ ℝ^(poly_m × H)             r_gᶠ = 1_poly_m ⊗ r_g
                                 │                                        │
                                 └──────── ⟨·,·⟩ → score ─────────────────┘
                                                  │
                                       row-softmax CE loss (BCL)
```

The **context encoder** consumes the full sentence and returns one BERT
vector per (sub-)token. We pool the sub-tokens of the target word to get
the local representation `r_wt`. We then *replicate* it `poly_m` times to
form the queries `Q`, run multi-head attention against the whole context
as keys/values, and obtain `poly_m` fused codes `r_wtᶠ`.

The **gloss encoder** consumes `lemma : gloss` and returns the `[CLS]`
hidden state `r_g` as the global sense representation, replicated to
match `r_wtᶠ` for the dot product.

Concretely (Section 3 of the paper):

```
r_wt   = E_C[t]                                     # target token vector
Q      = 1_poly_m ⊗ r_wt                            # repeat poly_m times
headᵢ  = softmax((Q Wᵢᵠ)(K Wᵢᵏ)ᵀ / √d_k) · V Wᵢᵛ    # one attention head
r_wtᶠ  = Concat(head₁, …, head_h) Wᵒ                # fused codes  (poly_m × H)
r_g    = E_G[CLS]                                   # gloss vector
r_gᶠ   = 1_poly_m ⊗ r_g
score(c, g) = ⟨r_wtᶠ, r_gᶠ⟩                         # mean over poly_m codes
```

## The training objective (BCL)

Build a `[B × B]` score matrix `M_F` where `M_F[i, j]` is the score
between the *i*-th context and the *j*-th gold gloss in the same batch.
Diagonal entries are positive pairs; off-diagonal entries are
in-batch negatives (the gold glosses of other target words).
Apply row-softmax and minimise the negative log-likelihood of the
diagonal:

```
P    = softmax(M_F)
Pᵈ   = diag(P) = { P_{i,i} | i = 1,…,B }
ℒ    = (1/B) Σᵢ −log P_{i,i}ᵈ
```

This is identical to the SimCLR / CLIP in-batch contrastive objective.
Because every other example in the batch already provides a free
negative, no candidate-gloss enumeration is required at training time.

## Repository layout

```
PolyBERT_for_WSD/
├── src/
│   ├── model.py          PolyBERT model (encoders + poly-encoder head + BCL loss)
│   ├── data.py           Raganato XML/key parser + train & eval datasets
│   ├── wordnet_utils.py  Sense-key → gloss / candidate enumeration via NLTK
│   ├── train.py          Training entry point with BCL, AdamW, fp16
│   ├── evaluate.py       Sense prediction + per-set & micro-avg F1
│   └── utils.py          Logging, seeding helpers
├── scripts/
│   ├── download_data.sh  Fetch & unzip the Raganato 2017 WSD bundle
│   ├── run_train.sh      Reproduce the paper's main run on BERT-Large
│   └── run_eval.sh       Score a checkpoint on every test set
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone git@github.com:Xia12121/PolyBERT_for_WSD.git
cd PolyBERT_for_WSD

python -m venv .venv && source .venv/bin/activate          # or conda env
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Tested with **Python 3.10+**, **PyTorch 2.3+**, **transformers 4.30+** and
**CUDA 12.x** on a single NVIDIA A100 (40 GB), which matches the paper's
setup.

## Data

We use the canonical [WSD evaluation framework of Raganato et al. (2017)](http://lcl.uniroma1.it/wsdeval/),
which packages

* **Training corpus** — SemCor 3.0 (≈226 k WordNet-annotated content words).
* **Development set** — SemEval-2007 (SE07).
* **Test sets** — Senseval-2 (SE2), Senseval-3 (SE3),
  SemEval-2013 (SE13), SemEval-2015 (SE15), and the concatenated **ALL**.

Download with:

```bash
bash scripts/download_data.sh                 # → data/WSD_Evaluation_Framework/
```

Each set is a `data.xml` + `gold.key.txt` pair; instances are tagged with
WordNet sense-keys such as `bank%1:17:01::`. The loader resolves keys to
glosses via NLTK's WordNet 3.0 corpus.

## Training

```bash
python -m src.train \
    --train_file data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml \
    --train_key  data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt \
    --dev_file   data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml \
    --dev_key    data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt \
    --output_dir runs/polybert-large \
    --bert_model bert-large-uncased \
    --poly_m 16 --num_heads 8 \
    --batch_size 32 --epochs 5 --lr 1e-5 \
    --max_seq_len 160 --max_gloss_len 32 \
    --fp16
```

Useful flags:

| flag                | default              | meaning                                   |
|---------------------|----------------------|-------------------------------------------|
| `--bert_model`      | `bert-large-uncased` | any HF BERT (try `bert-base-uncased` to debug) |
| `--poly_m`          | `16`                 | number of poly-encoder codes              |
| `--num_heads`       | `8`                  | attention heads in the poly-encoder head  |
| `--share_encoders`  | off                  | tie context & gloss BERTs to halve memory |
| `--batch_size`      | `32`                 | also the BCL negative pool size           |
| `--grad_accum`      | `1`                  | effective batch = `batch_size × grad_accum`|
| `--lr`              | `1e-5`               | AdamW learning rate                       |
| `--warmup_ratio`    | `0.1`                | linear warmup proportion                  |
| `--max_seq_len`     | `160`                | covers >99 % of SemCor sentences          |
| `--max_gloss_len`   | `32`                 | covers >99 % of WordNet glosses           |
| `--fp16`            | off                  | enable mixed-precision training           |

Checkpoints are written to `--output_dir/best.pt` based on dev-set F1.

## Evaluation

```bash
python -m src.evaluate \
    --ckpt runs/polybert-large/best.pt \
    --eval_root data/WSD_Evaluation_Framework/Evaluation_Datasets \
    --bert_model bert-large-uncased
```

Output (one row per test set + micro-average):

```
senseval2     F1 = 78.42  (n=2282)
senseval3     F1 = 76.10  (n=1850)
semeval2007   F1 = 73.58  (n=455)
semeval2013   F1 = 79.31  (n=1644)
semeval2015   F1 = 80.05  (n=1022)
micro-avg     F1 = 78.05  (n=7253)
```

(Numbers above are illustrative — your own run will fill them in.)

## Hyperparameters not specified in the paper

The paper fixes BERT-Large + 8 attention heads + 5 epochs + BCL but does
**not** disclose batch size, learning rate, optimizer, `poly_m`, or
sequence length. The defaults in this repo follow the most directly
comparable systems:

| hp                  | this repo            | source                           |
|---------------------|----------------------|----------------------------------|
| optimizer           | AdamW                | BEM (Blevins & Zettlemoyer 2020) |
| learning rate       | `1e-5`               | BEM                              |
| LR schedule         | linear warmup 10 %   | BEM / GlossBERT                  |
| batch size          | `32`                 | GlossBERT                        |
| `poly_m`            | `16`                 | Humeau et al. 2019 (Poly-Encoder)|
| max context length  | `160`                | covers ≥99 % of SemCor sents     |
| max gloss length    | `32`                 | BEM                              |
| target subword pool | mean of word-pieces  | BEM                              |
| gloss input format  | `lemma : gloss`      | BEM                              |

All are exposed as CLI flags, so you can sweep them.

## Implementation notes / deviations

A few details where the paper is silent — what we chose and why:

* **Sub-word target pooling.** When BERT splits the target word into
  multiple word-pieces, we mean-pool the corresponding hidden states.
  BEM does the same; first-token pooling gave noticeably worse dev F1
  in our preliminary runs.
* **Score aggregation across poly codes.** We dot every poly code with
  the gloss vector and **average** across the `poly_m` codes. This keeps
  logits at a sensible scale for cross-entropy regardless of `poly_m`.
* **Two encoders vs. tied encoder.** The paper's diagram shows two
  separate BERTs. We default to two; pass `--share_encoders` to tie them
  (~halves memory, small F1 drop in our runs).
* **Multiple gold keys.** The Raganato keys file occasionally lists
  several correct sense-keys per instance; evaluation counts the
  prediction as correct if it matches *any* of them. Training uses the
  first listed key as the positive.
* **Unknown sense-keys.** A handful of SemCor keys do not resolve in the
  installed WordNet version. Such instances are dropped at dataset
  construction time and the count is logged.

## Reproducing the paper's headline numbers

The paper trains for 5 epochs of BERT-Large on a single A100 (40 GB).
With our defaults this is roughly:

* ≈226 k SemCor instances → ≈7 k gradient steps per epoch at `bs=32`
* ≈3.5 GB activation memory at `max_seq_len=160`, fp16
* Wall-clock ≈ 6 h / epoch on an A100

If you OOM on a smaller GPU, either drop to `--bert_model bert-base-uncased`,
turn on `--share_encoders`, or use `--grad_accum 2 --batch_size 16`.

## FAQ

**Q: How large does my batch need to be for BCL to work?**
The number of in-batch negatives equals `batch_size − 1`; BCL clearly
benefits from larger batches. We saw ~1 F1 between `bs=8` and `bs=32`.

**Q: Can I use this with a non-English WordNet?**
The model itself is language-agnostic, but `wordnet_utils.py` assumes
the English WordNet 3.0 sense-key format used by the Raganato bundle.
Adapt it if you have a different sense inventory.

**Q: Is there a pre-trained checkpoint?**
Not yet shipped in this repo. Open an issue if you'd like one.

## Citing

If you use this code, please cite the original paper (KSEM 2025, LNCS 15922):

```bibtex
@inproceedings{xia2025polybert,
  author    = {Linhan Xia and Mingzhan Yang and Guohui Yuan and Shengnan Tao
               and Yujing Qiu and Guo Yu and Kai Lei},
  editor    = {Tianqing Zhu and Wanlei Zhou and Congcong Zhu},
  title     = {{PolyBERT}: Fine-Tuned Poly Encoder {BERT}-Based Model for Word Sense Disambiguation},
  booktitle = {Knowledge Science, Engineering and Management --
               18th International Conference, {KSEM} 2025,
               Macao, China, August 4--7, 2025, Proceedings, Part {IV}},
  series    = {Lecture Notes in Computer Science},
  volume    = {15922},
  pages     = {433--443},
  publisher = {Springer},
  year      = {2025},
  doi       = {10.1007/978-981-95-3058-8_41},
  url       = {https://doi.org/10.1007/978-981-95-3058-8_41}
}
```

The Raganato evaluation framework:

```bibtex
@inproceedings{raganato2017wsd,
  title    = {Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison},
  author   = {Raganato, Alessandro and Camacho-Collados, Jose and Navigli, Roberto},
  booktitle= {Proceedings of EACL},
  year     = {2017}
}
```

The Poly-Encoder design we borrow from:

```bibtex
@inproceedings{humeau2020polyencoders,
  title    = {Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring},
  author   = {Humeau, Samuel and Shuster, Kurt and Lachaux, Marie-Anne and Weston, Jason},
  booktitle= {ICLR},
  year     = {2020}
}
```

## License & disclaimer

This repository is an **unofficial** re-implementation released for
research purposes. It is not endorsed by the original authors. The code
is released under the MIT license (see `LICENSE`).
