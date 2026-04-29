# This file contains code adapted from the DeepSeek-V4 project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2025 DeepSeek

# Ported for llm-compressor: removed TP, FP8/FP4 hardcoding, KV cache,
# inference_mode. Uses standard nn.Linear and pure PyTorch ops.

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modeling.deepseekv4.kernel import (
    FP8Linear,
    bf16_index,
    dequant_fp4_weight,
    dequant_fp8_weight,
    hc_split_sinkhorn,
    sparse_attn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.size(-1),), self.weight.float(), self.eps).type_as(x)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    original_seq_len: int,
    theta: float = 10000.0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """Precompute YaRN-extended rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len:
        low = math.floor(dim * math.log(original_seq_len / (beta_fast * 2 * math.pi)) / (2 * math.log(theta)))
        high = math.ceil(dim * math.log(original_seq_len / (beta_slow * 2 * math.pi)) / (2 * math.log(theta)))
        smooth = (torch.arange(dim // 2, dtype=torch.float32) - low).clamp(0, 1) / (high - low)
        freqs = freqs / ((1 - smooth) * factor + smooth)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, conj: bool = False) -> None:
    """Apply rotary embeddings in-place."""
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).float())
    fc = freqs_cis.view(1, freqs_cis.size(0), *([1] * (xc.ndim - 3)), freqs_cis.size(-1))
    if conj:
        fc = fc.conj()
    xc = xc * fc
    x.copy_(torch.view_as_real(xc).flatten(-2).type_as(x))


# ---------------------------------------------------------------------------
# Compressor & Indexer (KV compression for sparse attention)
# ---------------------------------------------------------------------------

class Compressor(nn.Module):
    """Compresses KV via learned gated pooling over consecutive tokens."""

    def __init__(self, config: ModelConfig, compress_ratio: int = 4, head_dim: int = 512):
        super().__init__()
        self.dim = config.dim
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, config.norm_eps)

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> Optional[torch.Tensor]:
        """Prefill-only compression (no KV cache needed for oneshot)."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        should_compress = seqlen >= ratio
        if not should_compress:
            return None

        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder > 0:
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if overlap:
            kv = self.overlap_transform(kv, 0)
            score = self.overlap_transform(score, float("-inf"))

        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(dtype))

        freqs_cis_compress = freqs_cis[:cutoff:ratio]
        apply_rotary_emb(kv[..., -rd:], freqs_cis_compress)
        return kv


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention."""

    def __init__(self, config: ModelConfig, compress_ratio: int = 4):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio

        self.wq_b = FP8Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.compressor = Compressor(config, compress_ratio, self.head_dim)

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, freqs_cis: torch.Tensor, offset: int
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.size()

        # Compress KV for indexing
        kv_compress = self.compressor(x, freqs_cis)
        if kv_compress is None:
            return torch.zeros(bsz, seqlen, 0, dtype=torch.int32, device=x.device)

        n_compress = kv_compress.size(1)

        # Compute query for index selection
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim:], freqs_cis)

        # Score compressed positions
        scores = bf16_index(q, kv_compress)  # (b, s, n_compress)

        # Add learned weights
        w = self.weights_proj(x).transpose(-1, -2)  # (b, n_heads, s)
        # Combine: use scores to select top-k
        topk = min(self.index_topk, n_compress)
        _, indices = scores.topk(topk, dim=-1)
        indices = indices + offset
        return indices.int()


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, device: torch.device):
    """Build sliding window attention indices for prefill."""
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, offset: int, device: torch.device):
    """Build compressed KV attention indices for prefill."""
    matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratios[layer_id]
        self.eps = config.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = FP8Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = FP8Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = FP8Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = FP8Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
        )
        self.wo_b = FP8Linear(self.n_groups * config.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)
            else:
                self.indexer = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        win = self.window_size

        # Q projection
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV projection
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # Build attention indices
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, x.device)

        if self.compress_ratio:
            offset = kv.size(1)
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, freqs_cis, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, bsz, seqlen, offset, x.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

            # Compress KV and concatenate
            kv_compress = self.compressor(x, freqs_cis)
            if kv_compress is not None:
                kv = torch.cat([kv, kv_compress], dim=1)

        topk_idxs = topk_idxs.int()

        # Sparse attention
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # Output projection (grouped)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        # Dequant wo_a FP8 weight for grouped einsum
        wo_a_weight = dequant_fp8_weight(self.wo_a.weight, self.wo_a.scale, self.wo_a.block_size)
        wo_a_weight = wo_a_weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
        x = self.wo_b(o.flatten(2))
        return x


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class Gate(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.hash = layer_id < config.n_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(config.vocab_size, config.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Auto-dequant FP4/FP8 weights during loading."""
        for name in ("w1", "w2", "w3"):
            w_key = f"{prefix}{name}.weight"
            s_key = f"{prefix}{name}.scale"
            if w_key in state_dict and s_key in state_dict:
                weight = state_dict[w_key]
                scale = state_dict[s_key]
                # FP4 packed as I8: shape is (out, in//2)
                if weight.dtype in (torch.int8, torch.uint8):
                    state_dict[w_key] = dequant_fp4_weight(weight, scale)
                # FP8: shape is (out, in)
                elif weight.dtype == torch.float8_e4m3fn:
                    state_dict[w_key] = dequant_fp8_weight(weight, scale)
                # Remove scale — nn.Linear doesn't have it
                del state_dict[s_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate = Gate(layer_id, config)
        self.experts = nn.ModuleList([
            Expert(config.dim, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)
            for _ in range(config.n_routed_experts)
        ])
        self.shared_experts = Expert(config.dim, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten() if input_ids is not None else None)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


# ---------------------------------------------------------------------------
# Block with Hyper-Connections
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = config.norm_eps
        self.attn = Attention(layer_id, config)
        self.ffn = MoE(layer_id, config)
        self.attn_norm = RMSNorm(config.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, self.norm_eps)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # Precompute freqs_cis for this layer
        compress_ratio = config.compress_ratios[layer_id]
        if compress_ratio:
            original_seq_len = config.original_seq_len
            rope_theta = config.compress_rope_theta
        else:
            original_seq_len = 0
            rope_theta = config.rope_theta
        freqs_cis = precompute_freqs_cis(
            config.rope_head_dim, config.max_seq_len,
            original_seq_len, rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seqlen = x.size(1)
        freqs_cis = self.freqs_cis[:seqlen]

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, freqs_cis, input_ids)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


# ---------------------------------------------------------------------------
# HF wrappers
# ---------------------------------------------------------------------------

class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = ModelConfig
    base_model_prefix = ""  # no prefix — checkpoint keys have no "model." prefix
    _no_split_modules = ["Block"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\..*"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass  # weights loaded from checkpoint


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([Block(i, config) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # HC head parameters (top-level, matching checkpoint names)
        hc_dim = config.hc_mult * config.dim
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.config = config
        self.post_init()

    def hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce hc_mult copies to 1 via learned weighted sum."""
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config.norm_eps)
        mixes = F.linear(x, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, input_ids)
        h = self.hc_head_reduce(h)
        hidden_states = self.norm(h)
        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
