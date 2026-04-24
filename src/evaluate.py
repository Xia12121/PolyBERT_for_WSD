"""Evaluation: per-instance sense prediction + F1 across SE2/SE3/SE07/SE13/SE15."""
from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data import (WSDEvalDataset, eval_collate, find_eval_pairs, parse_raganato)
from .model import PolyBERT


@torch.no_grad()
def evaluate_dataset(model: PolyBERT, loader: DataLoader, device) -> float:
    """F1 = correct / total. Multiple gold keys per instance: any-match counts."""
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        # batch is a list of per-instance dicts (eval_collate)
        ctx_ids = torch.stack([b["ctx_ids"] for b in batch]).to(device)
        ctx_mask = torch.stack([b["ctx_mask"] for b in batch]).to(device)
        tgt_s = torch.stack([b["tgt_start"] for b in batch]).to(device)
        tgt_e = torch.stack([b["tgt_end"] for b in batch]).to(device)

        ctx_codes = model.encode_context(ctx_ids, ctx_mask, tgt_s, tgt_e)  # [B, m, H]

        for i, ex in enumerate(batch):
            g_ids = ex["g_ids"].to(device)
            g_mask = ex["g_mask"].to(device)
            g_vecs = model.encode_gloss(g_ids, g_mask)                    # [N, H]
            scores = model.score_candidates(ctx_codes[i], g_vecs)          # [N]
            pred_idx = int(scores.argmax().item())
            pred_key = ex["candidate_keys"][pred_idx]
            if pred_key in ex["gold_keys"]:
                correct += 1
            total += 1
    return correct / max(1, total)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_root", required=True,
                   help="…/WSD_Evaluation_Framework/Evaluation_Datasets")
    p.add_argument("--bert_model", default="bert-large-uncased")
    p.add_argument("--max_seq_len", type=int, default=160)
    p.add_argument("--max_gloss_len", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved = ckpt.get("args", {})
    poly_m = saved.get("poly_m", 16)
    num_heads = saved.get("num_heads", 8)
    share = saved.get("share_encoders", False)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = PolyBERT(args.bert_model, poly_m=poly_m,
                     num_heads=num_heads, share_encoders=share).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pairs = find_eval_pairs(args.eval_root)
    if not pairs:
        raise FileNotFoundError(f"No eval sets found under {args.eval_root}")

    results: Dict[str, float] = {}
    for name, (xml, key) in pairs.items():
        insts = parse_raganato(xml, key)
        ds = WSDEvalDataset(insts, tokenizer,
                            max_ctx_len=args.max_seq_len,
                            max_gloss_len=args.max_gloss_len)
        loader = DataLoader(ds, batch_size=args.eval_batch_size,
                            collate_fn=eval_collate, shuffle=False,
                            num_workers=args.num_workers)
        f1 = evaluate_dataset(model, loader, device)
        results[name] = f1
        print(f"{name:<14s} F1 = {f1*100:5.2f}  (n={len(ds)})")

    if "ALL" not in results:
        # micro-average over the five eval sets
        sets = [n for n in results if n != "ALL"]
        # weight by dataset size for a true micro-F1
        total_correct = total = 0
        for n in sets:
            xml, key = pairs[n]
            insts = parse_raganato(xml, key)
            ds = WSDEvalDataset(insts, tokenizer,
                                max_ctx_len=args.max_seq_len,
                                max_gloss_len=args.max_gloss_len)
            total += len(ds)
            total_correct += int(round(results[n] * len(ds)))
        if total:
            print(f"{'micro-avg':<14s} F1 = {total_correct/total*100:5.2f}  (n={total})")


if __name__ == "__main__":
    main()
