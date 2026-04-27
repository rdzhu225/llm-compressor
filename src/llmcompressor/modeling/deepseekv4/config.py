# This file contains code adapted from the DeepSeek-V4 project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2025 DeepSeek

from typing import Literal, Optional

from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        vocab_size: int = 129280,
        dim: int = 4096,
        moe_inter_dim: int = 2048,
        n_layers: int = 43,
        n_hash_layers: int = 3,
        n_mtp_layers: int = 1,
        n_heads: int = 64,
        # moe
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        n_activated_experts: int = 6,
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus",
        route_scale: float = 1.5,
        swiglu_limit: float = 10.0,
        # attention
        q_lora_rank: int = 1024,
        head_dim: int = 512,
        rope_head_dim: int = 64,
        norm_eps: float = 1e-6,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        window_size: int = 128,
        compress_ratios: tuple = (
            0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
        ),
        # yarn
        compress_rope_theta: float = 160000.0,
        original_seq_len: int = 65536,
        rope_theta: float = 10000.0,
        rope_factor: float = 16.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
        # index
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        # hyper-connections
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # inference
        max_batch_size: int = 4,
        max_seq_len: int = 4096,
        # HF compat
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.moe_inter_dim = moe_inter_dim
        self.n_layers = n_layers
        self.n_hash_layers = n_hash_layers
        self.n_mtp_layers = n_mtp_layers
        self.n_heads = n_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.swiglu_limit = swiglu_limit
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.norm_eps = norm_eps
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.window_size = window_size
        self.compress_ratios = list(compress_ratios)
        self.compress_rope_theta = compress_rope_theta
        self.original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # HF-standard aliases
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
