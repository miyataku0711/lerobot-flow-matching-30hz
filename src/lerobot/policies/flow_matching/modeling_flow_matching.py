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
"""Flow Matching Policy: Diffusion Policy's network trained with a flow-matching objective.

The image encoders, the 1D conditional U-Net, the observation queues and the chunking logic are reused from
`lerobot.policies.diffusion` as is; only the training objective (velocity regression instead of noise
prediction) and the sampler (forward Euler instead of a DDPM/DDIM scheduler) change. Because the submodules
keep Diffusion Policy's attribute names (`diffusion.rgb_encoder`, `diffusion.unet`), code that reaches into a
Diffusion Policy's network (e.g. the async_chunk rollout adapter) also applies here.

References:
    - Lipman et al., "Flow Matching for Generative Modeling" (https://huggingface.co/papers/2210.02747)
    - Liu et al., "Flow Straight and Fast: Rectified Flow" (https://huggingface.co/papers/2209.03003)
    - Black et al., "pi0: A Vision-Language-Action Flow Model" (https://huggingface.co/papers/2410.24164)
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..common.flow_matching import euler_integrate, sample_time_beta
from ..diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
)
from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_dtype_from_parameters
from .configuration_flow_matching import FlowMatchingConfig


class FlowMatchingPolicy(DiffusionPolicy):
    """
    Diffusion Policy ("Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
    https://huggingface.co/papers/2303.04137) with the DDPM objective replaced by flow matching, so that a few
    Euler steps are enough at inference time.

    `select_action`, `predict_action_chunk`, `forward` and `reset` are inherited from `DiffusionPolicy`.
    """

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(
        self,
        config: FlowMatchingConfig,
        **kwargs,
    ):
        # Skip DiffusionPolicy.__init__: it requires diffusers and builds a noise scheduler.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they hold the n latest observations and actions
        self._queues: dict[str, deque] | None = None

        # Named `diffusion` like in DiffusionPolicy so that the inherited methods and the weight names match.
        self.diffusion = FlowMatchingModel(config)

        self.reset()


class FlowMatchingModel(DiffusionModel):
    """`DiffusionModel` with a flow-matching loss and an Euler sampler.

    `_prepare_global_conditioning` and `generate_actions` are inherited.
    """

    def __init__(self, config: FlowMatchingConfig):
        # Skip DiffusionModel.__init__: it builds a noise scheduler from fields this config does not have.
        nn.Module.__init__(self)
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.num_inference_steps = config.num_inference_steps

    def velocity(self, x_t: Tensor, time: Tensor, global_cond: Tensor | None) -> Tensor:
        """Predicted velocity `noise - actions` at `x_t` for flow times `time` of shape (B,) in [0, 1]."""
        return self.unet(x_t, time.to(x_t.dtype) * self.config.time_embed_scale, global_cond=global_cond)

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior (x at t=1).
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        sample = euler_integrate(
            lambda x_t, time: self.velocity(x_t, time, global_cond), sample, self.num_inference_steps
        )

        if self.config.clip_sample:
            sample = sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        return sample

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Point on the straight path from the clean trajectory (t=0) to noise (t=1).
        trajectory = batch[ACTION]
        bsize = trajectory.shape[0]
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        time = self._sample_time(bsize, trajectory.device)
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * trajectory

        pred = self.velocity(x_t, time, global_cond)
        target = noise - trajectory

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            mask = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()

    def _sample_time(self, bsize: int, device) -> Tensor:
        if self.config.time_sampling == "uniform":
            return torch.rand(bsize, device=device)
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )
