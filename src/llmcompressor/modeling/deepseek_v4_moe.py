import torch
from typing import Optional

from llmcompressor.modeling.deepseekv4.model import MoE as OriginalMoE
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("MoE")
class CalibrationDeepseekV4MoE(MoECalibrationModule):
    """
    Calibration version of DeepSeek V4 MoE that sends all tokens to all experts.
    Note: V4's MoE forward takes (x, input_ids) because hash routing needs input_ids.
    """

    is_permanent = True

    def __init__(
        self,
        original: OriginalMoE,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.dim = original.dim
        self.n_routed_experts = original.n_routed_experts
        self.n_activated_experts = original.n_activated_experts
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten() if input_ids is not None else None)
        y = torch.zeros_like(x, dtype=torch.float32)

        expert_mask = torch.nn.functional.one_hot(
            indices, num_classes=self.n_routed_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for i in range(self.n_routed_experts):
            expert = self.experts[i]
            token_indices, weight_indices = torch.where(expert_mask[i])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                # Run ALL tokens through expert for calibration statistics
                expert_output = expert(x)
                if has_tokens:
                    expert_weights = weights[token_indices, weight_indices]
                    routed_output = expert_output[token_indices] * expert_weights[:, None]
                    y.index_add_(0, token_indices, routed_output)
            else:
                if has_tokens:
                    expert_input = x[token_indices]
                    expert_output = expert(expert_input)
                    expert_weights = weights[token_indices, weight_indices]
                    routed_output = expert_output * expert_weights[:, None]
                    y.index_add_(0, token_indices, routed_output)

        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
