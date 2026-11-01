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

"""Steer the sampling of a new chunk towards what the robot will already have executed (local change, 2026-10-22).

This is the inpainting view of Real-Time Chunking (Black et al. 2025, https://arxiv.org/abs/2506.07339):
the first ``d`` actions of the new chunk are executed from the previous chunk while it is being
computed, so they are frozen to it, and the following ones are pulled towards it with a weight that
decays to 0. Unlike a blender, a guide acts inside the denoising loop, so it needs an adapter that
exposes its sampler (``supports_guidance``). LeRobot's own RTC (``policies/rtc``) only handles
flow-matching policies; these guides cover both epsilon-prediction DDPM/DDIM samplers (Diffusion
Policy) and flow-matching Euler samplers (SmolVLA, Flow Matching Policy).

A guide gets a :class:`GuideTarget` in the model's (normalized) action space and two hooks that the
adapter's sampling loop calls at every denoising step: ``correct_eps`` / ``after_step`` for DDPM/DDIM
(``x_t = sqrt(a) x0 + sqrt(1 - a) eps``), ``correct_velocity`` / ``after_flow_step`` for flow matching
(``x_t = t * noise + (1 - t) * x0``, ``v = noise - x0``, time going from 1 to 0 as in LeRobot).
To add a guide, subclass :class:`Guide`, add a config registered with
``GuideConfig.register_subclass`` and return the guide from ``build``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import draccus
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig, RTCProcessor


@dataclass
class GuideTarget:
    """What the new chunk should agree with, aligned to the model's horizon rows.

    ``actions[j]`` (model space) is the action expected at row ``j``; ``weights[j]`` in [0, 1] is how
    strongly row ``j`` is tied to it (1 = frozen, 0 = free). Shapes ``(H, A)`` and ``(H,)``.
    """

    actions: torch.Tensor
    weights: torch.Tensor


class Guide(abc.ABC):
    needs_grad: bool = False

    def __init__(self, schedule: str) -> None:
        self._prefix = RTCProcessor(RTCConfig(prefix_attention_schedule=RTCAttentionSchedule(schedule.upper())))

    def prefix_weights(self, frozen: int, end: int, total: int) -> torch.Tensor:
        """1 for rows ``< frozen``, decaying to 0 at ``end``, 0 after (LeRobot RTC's prefix schedule)."""
        return self._prefix.get_prefix_weights(frozen, end, total)

    def correct_eps(self, x_t: torch.Tensor, eps_fn, alpha_bar: float, target: GuideTarget) -> torch.Tensor:
        """Return the (possibly corrected) noise prediction at ``x_t``. Default: no correction."""
        return eps_fn(x_t)

    def after_step(self, x_prev: torch.Tensor, alpha_bar_prev: float, target: GuideTarget) -> torch.Tensor:
        """Adjust the sample after the scheduler step. Default: unchanged."""
        return x_prev

    def correct_velocity(self, x_t: torch.Tensor, v_fn, t: float, target: GuideTarget) -> torch.Tensor:
        """Flow matching: return the (possibly corrected) velocity at ``x_t``. Default: no correction."""
        return v_fn(x_t)

    def after_flow_step(
        self, x_next: torch.Tensor, t_next: float, target: GuideTarget, noise: torch.Tensor
    ) -> torch.Tensor:
        """Flow matching: adjust the sample after the Euler step to ``t_next``. Default: unchanged.

        ``noise`` is the initial sample of this chunk (the ``t = 1`` end of every straight path).
        """
        return x_next


@dataclass
class GuideConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self) -> Guide | None: ...


@GuideConfig.register_subclass("none")
@dataclass
class NoGuideConfig(GuideConfig):
    def build(self) -> Guide | None:
        return None


class InpaintGuide(Guide):
    """RePaint-style replacement: after each step, blend the tied rows with the target noised to the same level.

    ``x <- w * (sqrt(a) * y + sqrt(1 - a) * z) + (1 - w) * x``. At the last step ``a = 1`` so frozen rows
    equal the target exactly. No gradients, so it costs almost nothing on top of the sampler.
    """

    def after_step(self, x_prev: torch.Tensor, alpha_bar_prev: float, target: GuideTarget) -> torch.Tensor:
        y = target.actions.to(x_prev.dtype)
        w = target.weights.to(x_prev.dtype)[:, None]
        noised = alpha_bar_prev**0.5 * y + (1.0 - alpha_bar_prev) ** 0.5 * torch.randn_like(y)
        return w * noised + (1.0 - w) * x_prev

    def after_flow_step(
        self, x_next: torch.Tensor, t_next: float, target: GuideTarget, noise: torch.Tensor
    ) -> torch.Tensor:
        """``x <- w * ((1 - t) * y + t * noise) + (1 - w) * x``: the tied rows follow the straight path from
        their own initial noise to the target, so at ``t = 0`` frozen rows equal the target exactly."""
        y = target.actions.to(x_next.dtype)
        w = target.weights.to(x_next.dtype)[:, None]
        on_path = (1.0 - t_next) * y + t_next * noise
        return w * on_path + (1.0 - w) * x_next


@GuideConfig.register_subclass("inpaint")
@dataclass
class InpaintGuideConfig(GuideConfig):
    schedule: str = "exp"  # zeros | ones | linear | exp (LeRobot RTC's prefix_attention_schedule)

    def build(self) -> Guide | None:
        return InpaintGuide(self.schedule)


class RTCGuide(Guide):
    """Pseudo-inverse guidance (PiGDM) on the clean-sample estimate, as LeRobot RTC does for flow matching.

    ``x0 = (x_t - sqrt(1 - a) * eps) / sqrt(a)``; ``g = J^T (w * (y - x0))`` with ``J = dx0/dx_t`` (one
    vector-Jacobian product through the denoiser). The noise prediction is corrected so that the
    clean-sample estimate moves by ``strength * sqrt(a) * g`` (about ``strength * w * (y - x0)``, since
    ``J ~ I / sqrt(a)``): ``eps' = eps - strength * a * g / sqrt(1 - a)``.

    PiGDM's own weight (``1 / r^2``, ``r^2 = 1 - a``) moves ``x0`` by about ``w * (y - x0) / a``, which
    overshoots by ~50x at the first DDIM steps (``a ~ 0.02``) and, once clipped like LeRobot RTC's
    ``max_guidance_weight``, is too weak at the last ones. Bounding the move of ``x0`` instead keeps the
    pull the same at every step. The VJP roughly doubles the cost of each denoising step.
    """

    needs_grad = True

    def __init__(self, schedule: str, strength: float) -> None:
        super().__init__(schedule)
        self.strength = strength

    def correct_eps(self, x_t: torch.Tensor, eps_fn, alpha_bar: float, target: GuideTarget) -> torch.Tensor:
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            eps = eps_fn(x)
            x0 = (x - (1.0 - alpha_bar) ** 0.5 * eps) / alpha_bar**0.5
            err = (target.actions.to(x0.dtype) - x0) * target.weights.to(x0.dtype)[:, None]
            (g,) = torch.autograd.grad(x0, x, err.detach())
        return (eps - self.strength * alpha_bar * g / max(1.0 - alpha_bar, 1e-6) ** 0.5).detach()

    def correct_velocity(self, x_t: torch.Tensor, v_fn, t: float, target: GuideTarget) -> torch.Tensor:
        """Flow matching: ``x0 = x_t - t * v``; ``v' = v - strength * g / t`` moves ``x0`` by ``strength * g``.

        LeRobot RTC (``policies/rtc``) uses PiGDM's weight here, clipped by ``max_guidance_weight``; this
        keeps the move of the clean-sample estimate bounded the same way as for DDPM/DDIM above.
        """
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            v = v_fn(x)
            x0 = x - t * v
            err = (target.actions.to(x0.dtype) - x0) * target.weights.to(x0.dtype)[:, None]
            (g,) = torch.autograd.grad(x0, x, err.detach())
        return (v - self.strength * g / max(t, 1e-6)).detach()


@GuideConfig.register_subclass("rtc")
@dataclass
class RTCGuideConfig(GuideConfig):
    schedule: str = "exp"
    # Fraction of the gap to the target that the clean-sample estimate closes at each step (tied rows).
    strength: float = 1.0

    def build(self) -> Guide | None:
        return RTCGuide(self.schedule, self.strength)
