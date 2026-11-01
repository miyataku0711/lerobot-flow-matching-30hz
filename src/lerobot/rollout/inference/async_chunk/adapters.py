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

"""The only place where the async-chunk engine touches a policy (local change, 2026-10-22).

An adapter turns the last ``n_obs_steps`` observation frames into one action chunk whose row 0 is
the action for the tick of the newest frame, in robot action space and the rollout's action-key
order. Everything policy-specific (observation history, pre/post-processing, the sampler, speed-ups)
lives here, so triggers, blenders and guides never see a policy and the policies are not modified.

- :class:`GenericAdapter`: any policy whose ``predict_action_chunk`` takes one observation (ACT, SmolVLA, ...).
- :class:`DiffusionAdapter`: Diffusion Policy. Stacks ``n_obs_steps`` frames itself and calls
  ``predict_action_chunk`` on that batch (its offline path, so the policy's own queues are unused), or
  runs the denoising loop itself when a guide is given.
- :class:`FlowMatchingAdapter`: Flow Matching Policy (Diffusion Policy's network). Same as
  :class:`DiffusionAdapter`, with the flow-matching Euler loop when a guide is given.
- :class:`SmolVLAAdapter`: SmolVLA. ``predict_action_chunk`` without a guide; with one, runs the
  flow-matching Euler loop itself (prefix KV cache once, then ``denoise_step`` per step).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, replace

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .guides import Guide, GuideTarget

logger = logging.getLogger(__name__)


@dataclass
class AccelConfig:
    """Inference speed-ups applied by the adapter. Measured on Diffusion Policy, GB10 (2026-10-22)."""

    # Keep the weights in fp16 instead of letting autocast cast them on every call (needs use_amp).
    half_weights: bool = False
    # Run conv2d image encoders in channels_last (Diffusion ResNet18 x3 at 480x640: 9.7 -> 6.5 ms).
    channels_last: bool = False
    # False runs the Diffusion U-Net (large Conv1d weights, batch 1) without cuDNN, whose NHWC kernels
    # re-layout every weight on every call. With half_weights: 9.1 -> 4.0 ms per denoising step.
    unet_cudnn: bool = True
    # SmolVLA: replay the expert pass of each denoising step from a CUDA graph. The step is bound by
    # launching ~1000 small kernels from the CPU (GB10: 6.8 -> 3.9 ms per step, identical results).
    # The VLM's prefix pass is replayed too (AGX Orin: 62 -> 14 ms per chunk, identical results).
    # Diffusion / Flow Matching: replay the U-Net. Without cuDNN it is bound by the CPU launching its
    # kernels on AGX Orin (15.0 -> 4.9 ms per step, identical results).
    cuda_graph: bool = False


class ChunkPolicyAdapter:
    """Base adapter: one observation frame in, ``predict_action_chunk`` out."""

    supports_guidance: bool = False

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        robot_type: str,
        device: str | None,
        accel: AccelConfig,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task = task
        self.robot_type = robot_type
        self.device = torch.device(device or "cpu")
        self.accel = accel
        names = list(dataset_features["action"]["names"])
        # Policy action order -> rollout action-key order (what get_action must return).
        self._perm = np.array([names.index(k) for k in ordered_action_keys])
        self._inv_perm = np.argsort(self._perm)
        self._affine: tuple[np.ndarray, np.ndarray] | None = None
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

    # -- properties a sampler-level guide needs -------------------------------------------------
    n_obs_steps: int = 1

    @property
    def horizon_offset(self) -> int:
        """Row ``j`` of the sampler's horizon is tick ``k_obs - horizon_offset + j``."""
        return 0

    @property
    def horizon(self) -> int:
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------------------------------
    def setup(self) -> None:
        """Apply speed-ups, probe the postprocessor and warm up (called once, from the inference thread)."""
        self._probe_affine()
        self._warm_up()

    def _warm_up(self, calls: int = 2) -> None:
        """Run the inference on blank frames so CUDA/cuDNN initialisation (~0.5 s) happens before the loop.

        Strategies only pre-warm the sync engine (and only with torch.compile), so without this the
        first chunk of a run is computed inside the timed loop and the first tick stalls.
        """
        frame = {}
        for key, ft in self.policy.config.input_features.items():
            if ft.type is FeatureType.VISUAL:
                c, h, w = ft.shape
                frame[key] = np.zeros((h, w, c), dtype=np.uint8)
            else:
                frame[key] = np.zeros(ft.shape, dtype=np.float32)
        for _ in range(calls):
            self.predict([frame] * self.n_obs_steps)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _amp(self):
        use_amp = self.device.type == "cuda" and getattr(self.policy.config, "use_amp", False)
        return torch.autocast(device_type=self.device.type) if use_amp else nullcontext()

    def _probe_affine(self) -> None:
        """Model space -> robot space is affine for MEAN_STD / MIN_MAX; recover it to map actions back."""
        dim = len(self._perm)
        with torch.no_grad():
            vals = [
                self.postprocessor(torch.full((1, 1, dim), v, device=self.device)).reshape(-1).float().cpu().numpy()
                for v in (0.0, 1.0, 0.5)
            ]
        offset, scale = vals[0], vals[1] - vals[0]
        if np.allclose(vals[2], offset + 0.5 * scale, rtol=1e-4, atol=1e-4) and np.all(np.abs(scale) > 1e-12):
            self._affine = (offset, scale)
        else:
            logger.warning("Action postprocessor is not affine: guides are disabled for this policy")

    def to_model_space(self, actions: np.ndarray) -> torch.Tensor:
        """Inverse of the postprocessor + reordering, for rows of ``(N, A)`` actions in rollout key order."""
        if self._affine is None:
            raise RuntimeError("postprocessor is not affine")
        offset, scale = self._affine
        raw = (actions[:, self._inv_perm] - offset) / scale
        return torch.as_tensor(raw, dtype=torch.float32, device=self.device)

    # -- inference ------------------------------------------------------------------------------
    def _prepare(self, frame: dict) -> dict:
        obs = prepare_observation_for_inference(copy(frame), self.device, self.task, self.robot_type)
        return self.preprocessor(obs)

    def _model_chunk(self, frames: list[dict], guide: Guide | None, target: GuideTarget | None) -> torch.Tensor:
        return self.policy.predict_action_chunk(self._prepare(frames[-1]))

    def predict(
        self, frames: list[dict], guide: Guide | None = None, target: GuideTarget | None = None
    ) -> np.ndarray:
        """``frames``: the last ``n_obs_steps`` observation frames, oldest first. Returns ``(T, A)``."""
        stream_ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        # no_grad, not inference_mode: under inference_mode autocast does not reuse its fp16 weight
        # copies and re-casts every weight on every call (Diffusion U-Net: 9.1 -> 15.3 ms per step).
        with stream_ctx, torch.no_grad(), self._amp():
            raw = self._model_chunk(frames, guide, target)
            processed = self.postprocessor(raw.float())
        out = processed.reshape(processed.shape[-2], processed.shape[-1]).float().cpu().numpy()
        return out[:, self._perm]


class GenericAdapter(ChunkPolicyAdapter):
    def setup(self) -> None:
        if self.accel.half_weights:
            self.policy.to(torch.float16)
        if self.accel.channels_last:
            self.policy.to(memory_format=torch.channels_last)
        super().setup()


class DiffusionAdapter(ChunkPolicyAdapter):
    supports_guidance = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.policy.config
        self.n_obs_steps = cfg.n_obs_steps
        self._keys = [*cfg.image_features, OBS_STATE] + ([OBS_ENV_STATE] if cfg.env_state_feature else [])
        if cfg.n_action_steps > cfg.horizon - cfg.n_obs_steps + 1:
            logger.warning(
                "n_action_steps=%d > horizon - n_obs_steps + 1 = %d: each chunk only has %d usable actions",
                cfg.n_action_steps,
                cfg.horizon - cfg.n_obs_steps + 1,
                cfg.horizon - cfg.n_obs_steps + 1,
            )

    @property
    def horizon_offset(self) -> int:
        return self.n_obs_steps - 1

    @property
    def horizon(self) -> int:
        return self.policy.config.horizon

    def setup(self) -> None:
        m = self.policy.diffusion
        if self.accel.half_weights:
            m.to(torch.float16)
        if self.accel.channels_last and m.config.image_features:
            m.rgb_encoder.to(memory_format=torch.channels_last)
        # The engine calls setup on every start: wrap the U-Net once (a graph wrapped in another graph would
        # replay inside the outer capture), and keep the captured graph for the next start.
        if not getattr(self, "_unet_wrapped", False):
            if not self.accel.unet_cudnn:
                unet_forward = m.unet.forward

                def forward_without_cudnn(*args, **kwargs):
                    with torch.backends.cudnn.flags(enabled=False):
                        return unet_forward(*args, **kwargs)

                m.unet.forward = forward_without_cudnn
            if self.accel.cuda_graph and self.device.type == "cuda":
                m.unet.forward = _UNetGraph(m.unet.forward)
            self._unet_wrapped = True
        super().setup()

    def _amp(self):
        # No cast cache with a graph: the graph would keep pointers to cached casts that are freed when the
        # autocast region ends (with half_weights there is nothing to cast anyway).
        if self.accel.cuda_graph and self.device.type == "cuda" and getattr(self.policy.config, "use_amp", False):
            return torch.autocast(device_type=self.device.type, cache_enabled=False)
        return super()._amp()

    def _batch(self, frames: list[dict]) -> dict[str, torch.Tensor]:
        frames = [frames[0]] * (self.n_obs_steps - len(frames)) + frames[-self.n_obs_steps :]
        obs = [self._prepare(f) for f in frames]
        return {k: torch.stack([o[k] for o in obs], dim=1) for k in self._keys}

    def _model_chunk(self, frames: list[dict], guide: Guide | None, target: GuideTarget | None) -> torch.Tensor:
        batch = self._batch(frames)
        if guide is None or target is None:
            self.policy.reset()  # empty queues -> predict_action_chunk uses the batch as given
            return self.policy.predict_action_chunk(batch)
        return self._guided_sample(batch, guide, target)

    def _guided_sample(self, batch: dict, guide: Guide, target: GuideTarget) -> torch.Tensor:
        """``DiffusionModel.generate_actions`` with the guide's hooks around each denoising step."""
        m = self.policy.diffusion
        cfg = self.policy.config
        if cfg.prediction_type != "epsilon":
            raise ValueError(f"guides need prediction_type=epsilon, got {cfg.prediction_type}")
        if cfg.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in cfg.image_features], dim=-4)
        global_cond = m._prepare_global_conditioning(batch)
        dtype = next(m.unet.parameters()).dtype
        sample = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], device=self.device, dtype=dtype)
        scheduler = m.noise_scheduler
        scheduler.set_timesteps(m.num_inference_steps)
        timesteps = list(scheduler.timesteps)
        alphas = scheduler.alphas_cumprod
        for i, t in enumerate(timesteps):
            tt = torch.full((1,), int(t), dtype=torch.long, device=self.device)

            def eps_fn(x, tt=tt):
                return m.unet(x, tt, global_cond=global_cond)

            alpha_bar = float(alphas[int(t)])
            eps = guide.correct_eps(sample, eps_fn, alpha_bar, target) if guide.needs_grad else eps_fn(sample)
            sample = scheduler.step(eps, t, sample).prev_sample
            alpha_prev = float(alphas[int(timesteps[i + 1])]) if i + 1 < len(timesteps) else 1.0
            sample = guide.after_step(sample, alpha_prev, target)
        start = self.horizon_offset
        return sample[:, start : start + cfg.n_action_steps]


class _UNetGraph:
    """``unet.forward`` replayed from a CUDA graph, for the Diffusion / Flow Matching U-Net.

    The graph is captured on the first call (the adapter's warm-up) and reads its inputs from static
    buffers. Calls with gradients enabled (the rtc guide differentiates through the U-Net), without a
    global condition, or with other input shapes / dtypes than captured run eagerly.
    """

    def __init__(self, forward) -> None:
        self.forward = forward
        self.key: tuple | None = None

    def __call__(self, x: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor | None = None):
        if torch.is_grad_enabled() or global_cond is None or x.device.type != "cuda":
            return self.forward(x, timestep, global_cond=global_cond)
        key = tuple((tuple(t.shape), t.dtype) for t in (x, timestep, global_cond))
        if self.key is None:
            self._capture(x, timestep, global_cond)
            self.key = key
        elif key != self.key:
            return self.forward(x, timestep, global_cond=global_cond)
        self.x.copy_(x)
        self.timestep.copy_(timestep)
        self.global_cond.copy_(global_cond)
        self.graph.replay()
        return self.out.clone()

    def _capture(self, x: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> None:
        self.x, self.timestep, self.global_cond = x.clone(), timestep.clone(), global_cond.clone()
        side = torch.cuda.Stream(x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up kernels / allocator outside the capture
                self.forward(self.x, self.timestep, global_cond=self.global_cond)
        torch.cuda.current_stream(x.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.forward(self.x, self.timestep, global_cond=self.global_cond)


class FlowMatchingAdapter(DiffusionAdapter):
    """Flow Matching Policy (``policies/flow_matching``): Diffusion Policy's network and observation history,
    so everything but the sampler (and the accel hooks on ``policy.diffusion``) is inherited."""

    def _guided_sample(self, batch: dict, guide: Guide, target: GuideTarget) -> torch.Tensor:
        """``FlowMatchingModel.generate_actions`` with the guide's flow hooks around each Euler step."""
        m = self.policy.diffusion
        cfg = self.policy.config
        if cfg.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in cfg.image_features], dim=-4)
        global_cond = m._prepare_global_conditioning(batch)
        dtype = next(m.unet.parameters()).dtype
        noise = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], device=self.device, dtype=dtype)
        x_t, dt = noise, -1.0 / m.num_inference_steps
        for step in range(m.num_inference_steps):
            t = 1.0 + step * dt
            time = torch.full((1,), t, dtype=torch.float32, device=self.device)

            def v_fn(x, time=time):
                return m.velocity(x, time, global_cond)

            v_t = guide.correct_velocity(x_t, v_fn, t, target) if guide.needs_grad else v_fn(x_t)
            x_t = guide.after_flow_step(x_t + dt * v_t, t + dt, target, noise)
        if cfg.clip_sample:
            x_t = x_t.clamp(-cfg.clip_sample_range, cfg.clip_sample_range)
        start = self.horizon_offset
        return x_t[:, start : start + cfg.n_action_steps]


class SmolVLAAdapter(GenericAdapter):
    supports_guidance = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph: _DenoiseGraph | None = None
        self._prefix_graph: _PrefixGraph | None = None

    @property
    def horizon(self) -> int:
        return self.policy.config.chunk_size

    def setup(self) -> None:
        if self.accel.half_weights:
            # SmolVLA mixes bf16 (text model, expert) and fp32 (projections, vision tower from scratch)
            # weights, and feeds fp32 tensors into several of them: an fp16 cast fails at the first call.
            logger.warning("--inference.accel.half_weights is not supported for SmolVLA: ignored")
            self.accel = replace(self.accel, half_weights=False)
        super().setup()

    def _amp(self):
        # bf16 as in training (use_amp trains in bf16 here), not autocast's fp16 default: the text model and
        # expert are bf16 already. No cast cache: a CUDA graph would keep pointers to cached casts that are
        # freed when the autocast region ends.
        if self.device.type == "cuda" and getattr(self.policy.config, "use_amp", False):
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, cache_enabled=False)
        return nullcontext()

    def _model_chunk(self, frames: list[dict], guide: Guide | None, target: GuideTarget | None) -> torch.Tensor:
        batch = self._prepare(frames[-1])
        if guide is None or target is None:
            guide = None
            if not self._use_graph(guide):
                return self.policy.predict_action_chunk(batch)
        return self._sample(batch, guide, target)

    def _use_graph(self, guide: Guide | None) -> bool:
        # A graph cannot be differentiated through: guides that need gradients run eagerly.
        return self.accel.cuda_graph and self.device.type == "cuda" and not (guide is not None and guide.needs_grad)

    def _sample(self, batch: dict, guide: Guide | None, target: GuideTarget | None) -> torch.Tensor:
        """``VLAFlowMatching.sample_actions``, with the guide's hooks around each Euler step if one is given.

        Without a guide and a graph this computes exactly what ``predict_action_chunk`` does (same noise).
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        policy, m, cfg = self.policy, self.policy.model, self.policy.config
        if cfg.adapt_to_pi_aloha:
            raise NotImplementedError("the SmolVLA adapter's sampler does not support adapt_to_pi_aloha")
        batch = policy._prepare_batch(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if self._use_graph(guide) and cfg.use_cache:
            if self._prefix_graph is None or not self._prefix_graph.matches(prefix_embs, prefix_att_2d):
                self._prefix_graph = _PrefixGraph(m, prefix_embs, prefix_att_2d, prefix_position_ids)
            past_key_values = self._prefix_graph(prefix_embs, prefix_att_2d, prefix_position_ids)
        else:
            _, past_key_values = m.vlm_with_expert.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=cfg.use_cache,
            )
        noise = m.sample_noise((1, cfg.chunk_size, cfg.max_action_dim), self.device)
        dim = cfg.action_feature.shape[0]
        if guide is not None:
            # The sampler runs on actions zero-padded to max_action_dim (as in training): pad the target too.
            padded = torch.zeros(cfg.chunk_size, cfg.max_action_dim, device=self.device)
            padded[:, : target.actions.shape[-1]] = target.actions
            target = GuideTarget(actions=padded, weights=target.weights.to(self.device))

        if self._use_graph(guide):
            if self._graph is None or not self._graph.matches(prefix_pad_masks, past_key_values):
                self._graph = _DenoiseGraph(m, prefix_pad_masks, past_key_values, noise)
            self._graph.load(prefix_pad_masks, past_key_values)
            denoise = self._graph
        else:

            def denoise(x, timestep):
                return m.denoise_step(
                    prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x, timestep=timestep
                )

        x_t, dt = noise, -1.0 / cfg.num_steps
        for step in range(cfg.num_steps):
            t = 1.0 + step * dt
            timestep = torch.tensor(t, dtype=torch.float32, device=self.device).expand(1)

            def v_fn(x, timestep=timestep):
                return denoise(x, timestep)

            if guide is None:
                x_t = x_t + dt * v_fn(x_t)
                continue
            v_t = guide.correct_velocity(x_t, v_fn, t, target) if guide.needs_grad else v_fn(x_t)
            x_t = guide.after_flow_step(x_t + dt * v_t, t + dt, target, noise)
        return x_t[:, :, :dim]


class _PrefixGraph:
    """The VLM's pass over the prefix (images, language, state) that fills the K/V cache, replayed from a CUDA graph.

    Like the expert pass, it is bound by the CPU launching its kernels on AGX Orin. ``embed_prefix`` stays
    eager: the vision embeddings index with a boolean mask (a device sync) and the attention mask is built
    from a Python list, neither of which can be captured; the vision tower behind them is bound by the GPU.
    The captured cache's tensors are overwritten by each replay; the returned cache object is pointed back
    at them first, since the denoise graph rebinds its entries.
    """

    def __init__(self, model, embs: torch.Tensor, att_2d: torch.Tensor, position_ids: torch.Tensor) -> None:
        self.m = model
        self.embs, self.att_2d, self.position_ids = embs.clone(), att_2d.clone(), position_ids.clone()
        side = torch.cuda.Stream(embs.device)
        side.wait_stream(torch.cuda.current_stream(embs.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up kernels / allocator outside the capture
                self._forward()
        torch.cuda.current_stream(embs.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.cache = self._forward()
        self.keys = [layer.keys for layer in self.cache.layers]
        self.values = [layer.values for layer in self.cache.layers]

    def _forward(self):
        _, past_key_values = self.m.vlm_with_expert.forward(
            attention_mask=self.att_2d,
            position_ids=self.position_ids,
            past_key_values=None,
            inputs_embeds=[self.embs, None],
            use_cache=True,
        )
        return past_key_values

    def matches(self, embs: torch.Tensor, att_2d: torch.Tensor) -> bool:
        return embs.shape == self.embs.shape and embs.dtype == self.embs.dtype and att_2d.shape == self.att_2d.shape

    def __call__(self, embs: torch.Tensor, att_2d: torch.Tensor, position_ids: torch.Tensor):
        self.embs.copy_(embs)
        self.att_2d.copy_(att_2d)
        self.position_ids.copy_(position_ids)
        self.graph.replay()
        for layer, k, v in zip(self.cache.layers, self.keys, self.values, strict=True):
            layer.keys, layer.values = k, v
        return self.cache


class _DenoiseGraph:
    """``VLAFlowMatching.denoise_step`` whose expert pass (all the layers, ~1000 kernels) replays a CUDA graph.

    ``embed_suffix`` builds a mask from a Python list on every call, which cannot be captured, so it stays
    eager; it is a handful of kernels. The graph reads the prefix K/V from static buffers, refreshed by
    :meth:`load` for each chunk. ``DynamicCache.update`` appends the suffix K/V with ``torch.cat`` and
    ``denoise_step`` crops it again, which only rebinds tensors on the Python side: the cache entries are
    pointed back at the static buffers after the pass, so every replay starts from the same prefix.
    A new prefix shape (e.g. a task with another number of tokens) needs a new graph (:meth:`matches`).
    """

    def __init__(self, model, prefix_pad_masks: torch.Tensor, past_key_values, x_t: torch.Tensor) -> None:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        self.m = model
        self.chunk_size = model.config.chunk_size
        self.cache = past_key_values
        self.keys = [layer.keys.clone() for layer in past_key_values.layers]
        self.values = [layer.values.clone() for layer in past_key_values.layers]
        self._point_cache_at_static()
        timestep = torch.ones(1, device=x_t.device)
        embs, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
        self.suffix_pad, self.suffix_att2d = suffix_pad, make_att_2d_masks(suffix_pad, suffix_att)
        self.prefix_pad = prefix_pad_masks.clone()
        self.mask = self._full_mask(self.prefix_pad)
        self.position_ids = self._position_ids(self.prefix_pad)
        self.embs = embs.clone()

        side = torch.cuda.Stream(x_t.device)
        side.wait_stream(torch.cuda.current_stream(x_t.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up kernels / allocator outside the capture
                self._expert()
        torch.cuda.current_stream(x_t.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._expert()

    def _point_cache_at_static(self) -> None:
        for layer, k, v in zip(self.cache.layers, self.keys, self.values, strict=True):
            layer.keys, layer.values = k, v

    def _full_mask(self, prefix_pad: torch.Tensor) -> torch.Tensor:
        b, s, p = prefix_pad.shape[0], self.suffix_pad.shape[1], prefix_pad.shape[1]
        return torch.cat([prefix_pad[:, None, :].expand(b, s, p), self.suffix_att2d], dim=2)

    def _position_ids(self, prefix_pad: torch.Tensor) -> torch.Tensor:
        return torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(self.suffix_pad, dim=1) - 1

    def _expert(self) -> torch.Tensor:
        outputs, _ = self.m.vlm_with_expert.forward(
            attention_mask=self.mask,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            inputs_embeds=[None, self.embs],
            use_cache=True,
        )
        self._point_cache_at_static()
        return self.m.action_out_proj(outputs[1][:, -self.chunk_size :].to(dtype=torch.float32))

    def matches(self, prefix_pad_masks: torch.Tensor, past_key_values) -> bool:
        return prefix_pad_masks.shape == self.prefix_pad.shape and all(
            layer.keys.shape == k.shape and layer.values.shape == v.shape
            for layer, k, v in zip(past_key_values.layers, self.keys, self.values, strict=True)
        )

    def load(self, prefix_pad_masks: torch.Tensor, past_key_values) -> None:
        for layer, k, v in zip(past_key_values.layers, self.keys, self.values, strict=True):
            k.copy_(layer.keys)
            v.copy_(layer.values)
        self.prefix_pad.copy_(prefix_pad_masks)
        self.mask.copy_(self._full_mask(self.prefix_pad))
        self.position_ids.copy_(self._position_ids(self.prefix_pad))

    def __call__(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        embs, _, _ = self.m.embed_suffix(x_t, timestep)
        self.embs.copy_(embs)
        self.graph.replay()
        return self.out.clone()


def make_adapter(policy: PreTrainedPolicy, **kwargs) -> ChunkPolicyAdapter:
    if policy.config.type == "diffusion":
        return DiffusionAdapter(policy, **kwargs)
    if policy.config.type == "flow_matching":
        return FlowMatchingAdapter(policy, **kwargs)
    if policy.config.type == "smolvla":
        return SmolVLAAdapter(policy, **kwargs)
    if getattr(policy.config, "n_obs_steps", 1) > 1:
        raise NotImplementedError(f"{policy.config.type} with n_obs_steps > 1 has no async-chunk adapter yet")
    return GenericAdapter(policy, **kwargs)
