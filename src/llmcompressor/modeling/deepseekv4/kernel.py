# This file contains code adapted from the DeepSeek-V4 project.
# Pure PyTorch replacements for tilelang kernels.
# Copyright (c) 2025 DeepSeek (MIT License)

import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """
    Split mixes into pre/post/comb and apply Sinkhorn normalization to comb.

    Args:
        mixes: (b, s, mix_hc) where mix_hc = (2 + hc_mult) * hc_mult
        hc_scale: (3,) scaling factors for pre, post, comb
        hc_base: (mix_hc,) bias terms
        hc_mult: number of hyper-connection copies
        sinkhorn_iters: number of Sinkhorn iterations
        eps: numerical stability epsilon

    Returns:
        pre: (b, s, hc_mult)
        post: (b, s, hc_mult)
        comb: (b, s, hc_mult, hc_mult)
    """
    hc = hc_mult

    # Split mixes into pre, post, comb sections
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])

    # comb: reshape from flat to (hc, hc) matrix
    comb_flat = mixes[..., 2*hc:] * hc_scale[2] + hc_base[2*hc:]
    comb = comb_flat.unflatten(-1, (hc, hc))

    # softmax along last dim + eps
    comb = F.softmax(comb, dim=-1) + eps

    # Sinkhorn normalization
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def bf16_index(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute index scores for sparse attention routing.

    output[b, m, n] = sum_over_h(ReLU(K[b, n, :] * Q[b, m, h, :]))

    Args:
        q: (b, m, h, d)
        k: (b, n, d)

    Returns:
        scores: (b, m, n)
    """
    # (b, m, h, d) * (b, 1, 1, d) -> (b, m, h, d) -> relu -> sum over h
    # Actually: Q[b,m,h,:] dot K[b,n,:] for each h, then relu, then sum over h
    # scores[b,m,n] = sum_h relu(sum_d Q[b,m,h,d] * K[b,n,d])
    scores = torch.einsum("bmhd,bnd->bmhn", q, k)
    scores = torch.relu(scores).sum(dim=2)
    return scores


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Sparse multi-head attention via index gathering.

    Args:
        q: (b, s, h, d)
        kv: (b, n, d) - KV cache
        attn_sink: (h,) - learnable attention sink bias per head
        topk_idxs: (b, s, topk) - indices into kv, -1 means invalid
        softmax_scale: attention scaling factor

    Returns:
        o: (b, s, h, d)
    """
    b, s, h, d = q.size()
    topk = topk_idxs.size(-1)

    # Clamp invalid indices to 0, we'll mask them later
    valid_mask = topk_idxs >= 0  # (b, s, topk)
    safe_idxs = topk_idxs.clamp(min=0)

    # Gather KV: (b, s, topk, d)
    # Expand indices for gathering from kv (b, n, d)
    gather_idxs = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, d)  # (b, s, topk, d)
    kv_expanded = kv.unsqueeze(1).expand(-1, s, -1, -1)  # (b, s, n, d)
    kv_gathered = torch.gather(kv_expanded, 2, gather_idxs)  # (b, s, topk, d)

    # Compute attention scores: (b, s, h, topk)
    attn = torch.einsum("bshd,bstd->bsht", q, kv_gathered) * softmax_scale

    # Mask invalid positions
    invalid_mask = ~valid_mask.unsqueeze(2)  # (b, s, 1, topk)
    attn = attn.masked_fill(invalid_mask, float("-inf"))

    # Add attention sink: extra "position" with learned bias per head
    # sink_score: (1, 1, h, 1)
    sink_score = attn_sink.view(1, 1, h, 1)
    # Concatenate sink score, then softmax, then drop sink column
    attn_with_sink = torch.cat([attn, sink_score.expand(b, s, -1, -1)], dim=-1)
    attn_weights = F.softmax(attn_with_sink, dim=-1)
    attn_weights = attn_weights[..., :topk]  # drop sink column

    # Weighted sum: (b, s, h, d)
    o = torch.einsum("bsht,bstd->bshd", attn_weights, kv_gathered)
    return o
