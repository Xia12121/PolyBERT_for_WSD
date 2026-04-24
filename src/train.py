"""Train PolyBERT with Batch Contrastive Learning."""
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import (WSDEvalDataset, WSDTrainDataset, eval_collate, parse_raganato,
                   train_collate)
from .evaluate import evaluate_dataset
from .model import PolyBERT, bcl_loss
from .utils import set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--train_key", required=True)
    p.add_argument("--dev_file", required=True)
    p.add_argument("--dev_key", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--bert_model", default="bert-large-uncased")
    p.add_argument("--poly_m", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--share_encoders", action="store_true",
                   help="Use a single BERT for both context and gloss (memory saver)")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=160)
    p.add_argument("--max_gloss_len", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log = setup_logger(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}  args={vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    log.info("Parsing training corpus…")
    train_inst = parse_raganato(args.train_file, args.train_key)
    log.info(f"  {len(train_inst):,} train instances")
    train_ds = WSDTrainDataset(train_inst, tokenizer,
                               max_ctx_len=args.max_seq_len,
                               max_gloss_len=args.max_gloss_len)
    log.info(f"  {len(train_ds):,} train examples after gloss filtering")

    log.info("Parsing dev corpus…")
    dev_inst = parse_raganato(args.dev_file, args.dev_key)
    dev_ds = WSDEvalDataset(dev_inst, tokenizer,
                            max_ctx_len=args.max_seq_len,
                            max_gloss_len=args.max_gloss_len)
    log.info(f"  {len(dev_ds):,} dev examples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_collate, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=eval_collate, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PolyBERT(args.bert_model, poly_m=args.poly_m,
                     num_heads=args.num_heads,
                     share_encoders=args.share_encoders).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = AdamW(grouped, lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(bar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.fp16):
                scores = model(batch)            # [B, B]
                loss = bcl_loss(scores) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                bar.set_postfix(loss=f"{running / (step + 1):.4f}",
                                lr=f"{sched.get_last_lr()[0]:.2e}")

        log.info(f"epoch {epoch} | mean train loss = {running / len(train_loader):.4f}")
        dev_f1 = evaluate_dataset(model, dev_loader, device)
        log.info(f"epoch {epoch} | dev F1 = {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            ckpt = os.path.join(args.output_dir, "best.pt")
            torch.save({"model": model.state_dict(),
                        "args": vars(args), "dev_f1": dev_f1}, ckpt)
            log.info(f"  ↑ new best — saved to {ckpt}")

    log.info(f"best dev F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
