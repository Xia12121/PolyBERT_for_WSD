"""Raganato 2017 WSD framework loader + PolyBERT collator.

A single training example == one (sentence, target-word, gold-sense-key) triple.
For evaluation we additionally need every candidate sense-key of the target
word, so the predictor can score them all.

The framework's directory layout is::

    WSD_Evaluation_Framework/
      Training_Corpora/SemCor/{semcor.data.xml, semcor.gold.key.txt}
      Evaluation_Datasets/<set>/{<set>.data.xml, <set>.gold.key.txt}

`<set>` ∈ {senseval2, senseval3, semeval2007, semeval2013, semeval2015, ALL}.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from lxml import etree
from torch.utils.data import Dataset

from .wordnet_utils import candidate_keys, gloss_for_key


# ---------------------------------------------------------------------------
# Raw parsing
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    iid: str                 # instance id from the XML (matches the gold key)
    sent_tokens: List[str]   # word-form tokens of the *whole sentence*
    target_idx: int          # index of the target word inside sent_tokens
    lemma: str
    pos: str                 # NOUN / VERB / ADJ / ADV
    gold_keys: List[str]     # 1+ sense-keys (filled at __init__ from key file)


def _read_key_file(path: str) -> Dict[str, List[str]]:
    """Map instance-id -> [sense-key, ...]. Multiple gold keys per instance
    are allowed; we treat any of them as a correct answer."""
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            out[parts[0]] = parts[1:]
    return out


def parse_raganato(xml_path: str, key_path: str) -> List[Instance]:
    """Parse one (data.xml, gold.key.txt) pair from the framework."""
    keys = _read_key_file(key_path)
    instances: List[Instance] = []

    # The XML files are large; iterparse lets us stream sentences one at a time.
    context = etree.iterparse(xml_path, events=("end",), tag="sentence")
    for _, sent in context:
        tokens: List[str] = []
        targets: List[tuple[int, str, str, str]] = []  # (idx, iid, lemma, pos)
        for child in sent:
            # Both <wf> and <instance> contribute a surface word to the sentence.
            text = (child.text or "").strip()
            if not text:
                continue
            idx = len(tokens)
            tokens.append(text)
            if child.tag == "instance":
                targets.append(
                    (idx, child.get("id"), child.get("lemma"), child.get("pos"))
                )
        for idx, iid, lemma, pos in targets:
            gold = keys.get(iid)
            if not gold:           # untagged instance — skip
                continue
            instances.append(
                Instance(iid=iid, sent_tokens=tokens, target_idx=idx,
                         lemma=lemma, pos=pos, gold_keys=gold)
            )
        sent.clear()
        # free already-processed XML siblings
        while sent.getprevious() is not None:
            del sent.getparent()[0]

    return instances


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def encode_context(tokenizer, sent_tokens: Sequence[str], target_idx: int,
                   max_len: int):
    """Tokenise a sentence keeping a record of which BERT subwords correspond
    to the target word. Returns: input_ids, attention_mask, (start, end_exclusive)
    indices into the BERT subword sequence (inclusive of [CLS] offset)."""
    pieces: List[int] = [tokenizer.cls_token_id]
    target_start: int | None = None
    target_end: int | None = None

    for i, tok in enumerate(sent_tokens):
        sub = tokenizer.encode(tok, add_special_tokens=False)
        if not sub:
            continue
        if i == target_idx:
            target_start = len(pieces)
        pieces.extend(sub)
        if i == target_idx:
            target_end = len(pieces)
        if len(pieces) >= max_len - 1:
            break
    pieces.append(tokenizer.sep_token_id)

    # Truncation may have dropped the target — clamp to the last legal slot.
    if target_start is None or target_start >= max_len - 1:
        target_start = max_len - 2
        target_end = max_len - 1
    target_end = min(target_end, max_len - 1)
    target_end = max(target_end, target_start + 1)

    pieces = pieces[:max_len]
    attn = [1] * len(pieces) + [0] * (max_len - len(pieces))
    pieces = pieces + [tokenizer.pad_token_id] * (max_len - len(pieces))
    return pieces, attn, target_start, target_end


def encode_gloss(tokenizer, lemma: str, gloss: str, max_len: int):
    """`lemma : gloss` packed as a single BERT input. Following BEM."""
    text = f"{lemma} : {gloss}"
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    padding="max_length", return_tensors=None)
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class WSDTrainDataset(Dataset):
    """One training example -> (context-input, gold-gloss-input). Negatives
    are constructed implicitly by Batch Contrastive Learning at the loss step."""

    def __init__(self, instances: Iterable[Instance], tokenizer,
                 max_ctx_len: int = 160, max_gloss_len: int = 32):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_gloss_len = max_gloss_len
        self.examples: List[tuple[Instance, str, str]] = []
        for inst in instances:
            gold_key = inst.gold_keys[0]
            gloss = gloss_for_key(gold_key)
            if gloss is None:        # unknown sense-key in this WordNet version
                continue
            self.examples.append((inst, gold_key, gloss))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        inst, _gold_key, gloss = self.examples[i]
        ctx_ids, ctx_mask, t_s, t_e = encode_context(
            self.tokenizer, inst.sent_tokens, inst.target_idx, self.max_ctx_len)
        g_ids, g_mask = encode_gloss(self.tokenizer, inst.lemma, gloss, self.max_gloss_len)
        return {
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "ctx_mask": torch.tensor(ctx_mask, dtype=torch.long),
            "tgt_start": torch.tensor(t_s, dtype=torch.long),
            "tgt_end": torch.tensor(t_e, dtype=torch.long),
            "g_ids": torch.tensor(g_ids, dtype=torch.long),
            "g_mask": torch.tensor(g_mask, dtype=torch.long),
        }


def train_collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


class WSDEvalDataset(Dataset):
    """Evaluation example carries every candidate sense for the target word
    so the model can score them all and we can compute F1."""

    def __init__(self, instances: Iterable[Instance], tokenizer,
                 max_ctx_len: int = 160, max_gloss_len: int = 32):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_gloss_len = max_gloss_len
        self.examples: List[tuple[Instance, List[str], List[str]]] = []
        for inst in instances:
            cand = list(candidate_keys(inst.lemma, inst.pos))
            # ensure each gold key is among the candidates (rare WN drift)
            for gk in inst.gold_keys:
                if gk not in cand:
                    cand.append(gk)
            cand_glosses = [gloss_for_key(k) or "" for k in cand]
            keep = [(k, g) for k, g in zip(cand, cand_glosses) if g]
            if not keep:
                continue
            cks, cgs = zip(*keep)
            self.examples.append((inst, list(cks), list(cgs)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        inst, cks, cgs = self.examples[i]
        ctx_ids, ctx_mask, t_s, t_e = encode_context(
            self.tokenizer, inst.sent_tokens, inst.target_idx, self.max_ctx_len)
        g_ids, g_mask = [], []
        for gloss in cgs:
            ids, mask = encode_gloss(self.tokenizer, inst.lemma, gloss, self.max_gloss_len)
            g_ids.append(ids)
            g_mask.append(mask)
        return {
            "iid": inst.iid,
            "gold_keys": inst.gold_keys,
            "candidate_keys": cks,
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "ctx_mask": torch.tensor(ctx_mask, dtype=torch.long),
            "tgt_start": torch.tensor(t_s, dtype=torch.long),
            "tgt_end": torch.tensor(t_e, dtype=torch.long),
            "g_ids": torch.tensor(g_ids, dtype=torch.long),     # [n_cand, L]
            "g_mask": torch.tensor(g_mask, dtype=torch.long),
        }


def eval_collate(batch):
    """Eval batch keeps lists for variable-length per-instance candidate sets."""
    return batch


# ---------------------------------------------------------------------------
# Convenience: discover the standard eval sets in the Raganato bundle
# ---------------------------------------------------------------------------

EVAL_SETS = ("senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015")


def find_eval_pairs(eval_root: str) -> Dict[str, tuple[str, str]]:
    """Return {set_name: (data_xml, gold_key)} for every known eval set
    that exists under `eval_root`."""
    pairs: Dict[str, tuple[str, str]] = {}
    for name in EVAL_SETS:
        d = os.path.join(eval_root, name)
        x = os.path.join(d, f"{name}.data.xml")
        k = os.path.join(d, f"{name}.gold.key.txt")
        if os.path.exists(x) and os.path.exists(k):
            pairs[name] = (x, k)
    # ALL set is bundled too in some releases
    all_d = os.path.join(eval_root, "ALL")
    if os.path.exists(os.path.join(all_d, "ALL.data.xml")):
        pairs["ALL"] = (
            os.path.join(all_d, "ALL.data.xml"),
            os.path.join(all_d, "ALL.gold.key.txt"),
        )
    return pairs
