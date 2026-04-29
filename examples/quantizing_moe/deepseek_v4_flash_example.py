"""
DeepSeek V4 Flash quantization example.

The original checkpoint stores:
  - Attention layers (wq_a, wq_b, wkv, wo_a, wo_b) as FP8 (E4M3 + E8M0 scale)
  - Routed experts (w1, w2, w3) as FP4 (packed I8 + E8M0 scale)
  - Shared experts as FP8
  - Other layers (gate, compressor, norms, embed, head) as BF16/FP32

This model definition:
  - Keeps attention layers in native FP8 (using tilelang fp8_gemm)
  - Auto-dequants expert FP4/FP8 weights to BF16 during loading
  - Allows flexible quantization of any BF16 layer via oneshot

Example below: quantize routed + shared expert layers to INT4 (GPTQ W4A16).
Adjust config_groups and ignore list to target different layers.

Prerequisites:
  pip install tilelang

Usage:
  python deepseek_v4_flash_example.py
"""

import torch
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modeling.deepseekv4.model import DeepseekV4ForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier

# Register custom model with HF
AutoConfig.register("deepseek_v4", ModelConfig)
AutoModelForCausalLM.register(ModelConfig, DeepseekV4ForCausalLM)

# Path to the original checkpoint (FP8/FP4 weights loaded directly)
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
SAVE_DIR = "DeepSeek-V4-Flash-W4A16"

model = DeepseekV4ForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Calibration settings
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Quantization recipe: MoE experts → GPTQ W4A16 (INT4 weights)
# Attention layers stay in original FP8 (handled by FP8Linear).
# Adjust targets/ignore to quantize different layers.
recipe = QuantizationModifier(
    config_groups={
        "moe_experts": QuantizationScheme(
            targets=[
                # Routed experts (dequanted from FP4 to BF16 during load)
                r"re:.*ffn\.experts\.\d+\.(w1|w2|w3)$",
                # Shared experts (dequanted from FP8 to BF16 during load)
                r"re:.*ffn\.shared_experts\.(w1|w2|w3)$",
            ],
            weights={
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
        ),
    },
    # Don't quantize these (they stay in their original format)
    ignore=["head", "gate", "compressor", "indexer"],
)

oneshot(
    model=model,
    processor=tokenizer,
    dataset="ultrachat-200k",
    splits={"calibration": f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
