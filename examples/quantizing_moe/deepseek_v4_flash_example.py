"""
DeepSeek V4 Flash quantization example.

Prerequisites:
  1. The original checkpoint uses FP8/FP4 weights. You must first convert
     them to bfloat16 before running this script. Use the convert.py from
     the model repo or compressed_tensors FP8BlockDequantizer.
  2. Register the custom model before loading.

Usage:
  python deepseek_v4_flash_example.py
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.deepseekv4.config import ModelConfig
from llmcompressor.modeling.deepseekv4.model import DeepseekV4ForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier

# Register custom model with HF
AutoConfig.register("deepseek_v4", ModelConfig)
AutoModelForCausalLM.register(ModelConfig, DeepseekV4ForCausalLM)

# Path to the bfloat16-converted checkpoint
MODEL_ID = "DeepSeek-V4-Flash-bf16"
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

# GPTQ W4A16 quantization
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["head", "gate"],  # don't quantize lm_head or MoE gate
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
