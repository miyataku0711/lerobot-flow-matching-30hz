#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig

from ..diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    """Configuration class for FlowMatchingPolicy.

    Diffusion Policy's network (ResNet + SpatialSoftmax image encoders and the 1D conditional U-Net) trained
    with a flow-matching objective instead of DDPM. The observation, U-Net and optimization fields carry the
    same names and defaults as `DiffusionConfig` (see its docstring), so the two policies differ only in the
    objective and the sampler.

    The flow follows the openpi / LeRobot convention shared with pi0 and SmolVLA
    (`lerobot.policies.common.flow_matching`): `x_t = t * noise + (1 - t) * actions`, the network predicts the
    velocity `noise - actions`, and sampling integrates from t=1 (noise) to t=0 (actions) with forward Euler.

    Args:
        num_inference_steps: Number of Euler steps at inference time.
        time_sampling: How training times are drawn. "beta" uses openpi's
            `Beta(alpha, beta) * scale + offset`, which puts more weight near t=1 (noisy samples).
            "uniform" draws t uniformly in [0, 1).
        time_sampling_beta_alpha: Alpha of the Beta distribution ("beta" only).
        time_sampling_beta_beta: Beta of the Beta distribution ("beta" only).
        time_sampling_scale: Scale applied to the Beta sample ("beta" only).
        time_sampling_offset: Offset added after scaling ("beta" only).
        time_embed_scale: The U-Net's sinusoidal step embedding is fed `t * time_embed_scale`. 100 matches the
            0..99 integer range Diffusion Policy trains its embedding on.
        clip_sample: Whether to clip the final sample to [-`clip_sample_range`, +`clip_sample_range`].
            Only the output of the integration is clipped (DDPM clips at every step).
        clip_sample_range: The magnitude of the clipping range as described above.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling (same as DiffusionConfig).
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

    # Flow matching.
    num_inference_steps: int = 10
    time_sampling: str = "beta"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    time_embed_scale: float = 100.0
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_time_samplings = ["beta", "uniform"]
        if self.time_sampling not in supported_time_samplings:
            raise ValueError(
                f"`time_sampling` must be one of {supported_time_samplings}. Got {self.time_sampling}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive. Got {self.num_inference_steps}.")

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        DiffusionConfig.validate_features(self)

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
