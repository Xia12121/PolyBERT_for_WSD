"""PolyBERT model.

Two BERT encoders (context, gloss) + a poly-encoder head that fuses
token-level (target word) and sequence-level (whole-context) semantics
via multi-head attention.

Notation follows Section 3 of the paper:

    r_wt        = E_C[t]              (target-word token vector — local)
    Q           = 1_m ⊗ r_wt          (replicate poly_m times)
    head_i      = softmax(QW_i^Q (KW_i^K)^T / √d_k) V W_i^V
    r_wt^F      = Concat(head_1..h) W^O
    r_g         = E_G[CLS]            (gloss [CLS] — global)
    r_g^F       = 1_m ⊗ r_g
    score(c,g)  = ⟨r_wt^F, r_g^F⟩  / poly_m   (sum then normalise)

For Batch Contrastive Learning, the score-matrix M_F = R_wt R_g^T is a
[B × B] matrix whose diagonal holds the gold pairs; row-softmax CE is the
loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PolyEncoderHead(nn.Module):
    """Multi-head attention that fuses the replicated target-word query
    with the whole context as keys/values."""

    def __init__(self, hidden: int, num_heads: int = 8, poly_m: int = 16):
        super().__init__()
        assert hidden % num_heads == 0, "hidden must be divisible by num_heads"
        self.hidden = hidden
        self.num_heads = num_heads
        self.poly_m = poly_m

        self.W_Q = nn.Linear(hidden, hidden, bias=False)
        self.W_K = nn.Linear(hidden, hidden, bias=False)
        self.W_V = nn.Linear(hidden, hidden, bias=False)
        self.W_O = nn.Linear(hidden, hidden, bias=False)

    def forward(self, r_wt: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        """
        r_wt:     [B, H]              local target-word vector
        ctx:      [B, L, H]           full context token vectors
        ctx_mask: [B, L]              1 for real tokens, 0 for padding
        returns:  [B, poly_m, H]      fused poly-codes
        """
        B, L, H = ctx.shape
        m, h = self.poly_m, self.num_heads
        d_k = H // h

        # Q = 1_m ⊗ r_wt  ->  [B, m, H]
        Q = r_wt.unsqueeze(1).expand(B, m, H).contiguous()

        Q = self.W_Q(Q).view(B, m, h, d_k).transpose(1, 2)        # [B, h, m, d_k]
        K = self.W_K(ctx).view(B, L, h, d_k).transpose(1, 2)      # [B, h, L, d_k]
        V = self.W_V(ctx).view(B, L, h, d_k).transpose(1, 2)      # [B, h, L, d_k]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_k ** 0.5)   # [B,h,m,L]
        scores = scores.masked_fill(ctx_mask[:, None, None, :] == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                    # [B,h,m,d_k]
        out = out.transpose(1, 2).contiguous().view(B, m, H)
        return self.W_O(out)


def pool_target(token_states: torch.Tensor, start: torch.Tensor,
                end: torch.Tensor) -> torch.Tensor:
    """Mean-pool the BERT subword vectors that make up the target word.

    token_states: [B, L, H]
    start, end:   [B]   (end is exclusive)
    """
    B, L, H = token_states.shape
    pos = torch.arange(L, device=token_states.device).unsqueeze(0).expand(B, L)
    mask = (pos >= start.unsqueeze(1)) & (pos < end.unsqueeze(1))   # [B, L]
    mask = mask.unsqueeze(-1).float()
    summed = (token_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class PolyBERT(nn.Module):
    def __init__(self, bert_model: str = "bert-large-uncased",
                 poly_m: int = 16, num_heads: int = 8,
                 share_encoders: bool = False):
        super().__init__()
        self.context_bert = AutoModel.from_pretrained(bert_model)
        self.gloss_bert = (
            self.context_bert if share_encoders
            else AutoModel.from_pretrained(bert_model)
        )
        hidden = self.context_bert.config.hidden_size
        self.poly_m = poly_m
        self.head = PolyEncoderHead(hidden, num_heads=num_heads, poly_m=poly_m)

    # -- single-side forwards (handy for evaluation caching) -----------------

    def encode_context(self, ctx_ids, ctx_mask, tgt_start, tgt_end):
        out = self.context_bert(input_ids=ctx_ids, attention_mask=ctx_mask)
        ctx = out.last_hidden_state                       # [B, L, H]
        r_wt = pool_target(ctx, tgt_start, tgt_end)       # [B, H]
        r_wtF = self.head(r_wt, ctx, ctx_mask)            # [B, m, H]
        return r_wtF

    def encode_gloss(self, g_ids, g_mask):
        out = self.gloss_bert(input_ids=g_ids, attention_mask=g_mask)
        return out.last_hidden_state[:, 0]                # [B, H]

    # -- training / scoring --------------------------------------------------

    def forward(self, batch):
        """Used during training. Returns the [B,B] BCL score matrix."""
        ctx_codes = self.encode_context(
            batch["ctx_ids"], batch["ctx_mask"],
            batch["tgt_start"], batch["tgt_end"])           # [B, m, H]
        g_vecs = self.encode_gloss(batch["g_ids"], batch["g_mask"])  # [B, H]
        # M_F[i, j] = <ctx_codes[i], g_vecs[j]> averaged over m codes
        # (averaging keeps the logits in a sensible scale for softmax CE).
        scores = torch.einsum("bmh,nh->bmn", ctx_codes, g_vecs)      # [B, m, B]
        return scores.mean(dim=1)                                    # [B, B]

    def score_candidates(self, ctx_codes: torch.Tensor,
                         g_vecs: torch.Tensor) -> torch.Tensor:
        """ctx_codes: [m, H]  (one instance), g_vecs: [N, H]
        returns the score for each of the N candidate glosses."""
        return torch.einsum("mh,nh->mn", ctx_codes, g_vecs).mean(dim=0)  # [N]


def bcl_loss(score_matrix: torch.Tensor) -> torch.Tensor:
    """Batch Contrastive Loss = row-softmax CE against the diagonal."""
    targets = torch.arange(score_matrix.size(0), device=score_matrix.device)
    return F.cross_entropy(score_matrix, targets)
