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

"""Tests for the async-chunk inference engine and its plug-ins (local change, 2026-10-22)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from lerobot.rollout.inference.async_chunk import AsyncChunkEngine, Chunk, TickClock
from lerobot.rollout.inference.async_chunk.adapters import AccelConfig
from lerobot.rollout.inference.async_chunk.blenders import (
    _AGGREGATE_FUNCTIONS,
    AggregateBlender,
    CrossfadeBlender,
    EnsembleBlender,
    SmoothBlender,
    SwitchBlender,
)
from lerobot.rollout.inference.async_chunk.guides import GuideTarget, InpaintGuide, RTCGuide
from lerobot.rollout.inference.async_chunk.triggers import ContinuousTrigger, PeriodTrigger, RemainingTrigger


def make_chunk(cid: int, k_obs: int, values: np.ndarray, k_ready: int | None = None) -> Chunk:
    actions = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return Chunk(id=cid, k_obs=k_obs, actions=actions, t_obs=0.0, t_start=0.0, t_ready=0.0,
                 k_ready=k_obs if k_ready is None else k_ready)


# -- timeline / triggers --------------------------------------------------------------------------


def test_tick_clock_follows_loop_clock():
    dt = 1 / 30
    clock = TickClock(fps=30)
    assert clock.tick(100.0) == 0
    assert clock.tick(100.0 + 0.010) == 0  # same period
    # First tick stalled 50 ms: the next iteration starts 1.6 periods in, but LoopClock catches up on
    # its schedule (late by less than a period), so it is tick 1 and the following one tick 2.
    assert clock.tick(100.0 + 0.053) == 1
    assert clock.tick(100.0 + 2 * dt + 0.003) == 2
    assert clock.tick(100.0 + 3 * dt + 0.001) == 3
    # More than a full period late: LoopClock re-anchors, the ticks that went by are skipped.
    assert clock.tick(100.0 + 3 * dt + 0.140) == 7
    assert clock.tick(100.0 + 3 * dt + 0.140 + dt) == 8
    clock.reset()
    assert clock.tick(5.0) == 0


def test_triggers():
    c = make_chunk(0, k_obs=10, values=np.zeros(20))  # covers ticks 10..29
    assert RemainingTrigger(5).should_start(0, None, None)
    assert not RemainingTrigger(5).should_start(24, c, 10)
    assert RemainingTrigger(5).should_start(25, c, 10)
    assert not PeriodTrigger(8).should_start(17, c, 10)
    assert PeriodTrigger(8).should_start(18, c, 10)
    assert ContinuousTrigger().should_start(11, c, 10)


# -- blenders --------------------------------------------------------------------------------------


def test_switch_uses_newest_chunk_by_tick():
    b = SwitchBlender()
    b.add(make_chunk(0, 0, np.arange(10)), k_next=0)
    assert b.action(3) == 3
    # Second chunk predicted from tick 3, arrived when tick 5 was next: its rows 0-1 are stale.
    b.add(make_chunk(1, 3, 100 + np.arange(3, 13)), k_next=5)
    assert b.action(5) == 105
    assert b.action(12) == 112
    assert b.action(13) is None


def reference_lerobot_async(queue: dict[int, float], incoming: dict[int, float], latest: int, fn) -> dict:
    """``robot_client._aggregate_action_queues`` with timestep -> action dicts."""
    out = {}
    for t, a in incoming.items():
        if t <= latest:
            continue
        out[t] = a if t not in queue else fn(queue[t], a)
    return out


@pytest.mark.parametrize("name", list(_AGGREGATE_FUNCTIONS))
def test_aggregate_blenders_match_lerobot_async_inference(name):
    fn = _AGGREGATE_FUNCTIONS[name]
    rng = np.random.default_rng(0)
    b = AggregateBlender(fn)
    ref: dict[int, float] = {}
    latest = -1
    for cid, (k_obs, k_next) in enumerate([(0, 0), (6, 8), (13, 15), (20, 22)]):
        vals = rng.normal(size=12)
        b.add(make_chunk(cid, k_obs, vals), k_next=k_next)
        latest = k_next - 1
        ref = reference_lerobot_async(ref, {k_obs + i: v for i, v in enumerate(vals)}, latest, fn)
        assert set(b.queue) == set(ref)
        for t in ref:
            assert b.action(t) == pytest.approx(ref[t])


def test_crossfade_moves_linearly():
    b = CrossfadeBlender(ticks=4)
    b.add(make_chunk(0, 0, np.zeros(20)), k_next=0)
    b.add(make_chunk(1, 2, np.ones(20), k_ready=5), k_next=5)
    got = [float(b.action(k)[0]) for k in range(5, 10)]
    assert got == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])
    assert float(b.action(4)[0]) == 0.0


def test_ensemble_weights_favour_older_chunks():
    b = EnsembleBlender(coeff=0.5, max_chunks=8)
    b.add(make_chunk(0, 0, np.zeros(20)), k_next=0)
    b.add(make_chunk(1, 2, np.ones(20)), k_next=3)
    w0, w1 = 1.0, math.exp(-0.5)
    assert float(b.action(4)[0]) == pytest.approx(w1 / (w0 + w1))


def test_smooth_removes_alternation_without_lag():
    b = SmoothBlender(window=3)
    ramp = np.arange(12, dtype=float)
    jitter = ramp + np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
    b.add(make_chunk(0, 0, jitter), k_next=0)
    got = np.array([b.action(k)[0] for k in range(1, 11)])
    # A centred average keeps the ramp where it is (no lag) and shrinks the +-1 alternation to 1/3
    # (with the opposite sign: the two neighbours outweigh the centre).
    assert np.allclose(got, ramp[1:11] + np.where(np.arange(1, 11) % 2 == 0, -1 / 3, 1 / 3))


# -- guides ----------------------------------------------------------------------------------------


def test_inpaint_freezes_prefix_at_last_step():
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    torch.manual_seed(0)
    sched = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    sched.set_timesteps(10)
    guide = InpaintGuide("exp")
    horizon, dim = 16, 3
    weights = guide.prefix_weights(4, 10, horizon)
    assert torch.all(weights[:4] == 1) and torch.all(weights[10:] == 0)
    target = GuideTarget(actions=torch.full((horizon, dim), 0.7), weights=weights)
    x = torch.randn(1, horizon, dim)
    ts = list(sched.timesteps)
    for i, t in enumerate(ts):
        x = sched.step(torch.zeros_like(x), t, x).prev_sample
        a_prev = float(sched.alphas_cumprod[int(ts[i + 1])]) if i + 1 < len(ts) else 1.0
        x = guide.after_step(x, a_prev, target)
    assert torch.allclose(x[0, :4], torch.full((4, dim), 0.7))
    assert not torch.allclose(x[0, 12:], torch.full((4, dim), 0.7))


def test_flow_inpaint_freezes_prefix_at_last_step():
    torch.manual_seed(0)
    guide = InpaintGuide("exp")
    horizon, dim, steps = 16, 3, 10
    weights = guide.prefix_weights(4, 10, horizon)
    target = GuideTarget(actions=torch.full((horizon, dim), 0.7), weights=weights)
    noise = torch.randn(1, horizon, dim)
    x, dt = noise, -1.0 / steps
    for step in range(steps):
        t = 1.0 + step * dt
        v = noise - torch.zeros_like(x)  # a model that always predicts x0 = 0
        x = guide.after_flow_step(x + dt * v, t + dt, target, noise)
    assert torch.allclose(x[0, :4], torch.full((4, dim), 0.7), atol=1e-6)
    assert torch.allclose(x[0, 10:], torch.zeros(6, dim), atol=1e-6)


def test_flow_rtc_moves_clean_estimate_of_tied_rows_only():
    torch.manual_seed(0)
    guide = RTCGuide("zeros", strength=0.5)
    horizon, dim, t = 8, 2, 0.6
    weights = guide.prefix_weights(3, 3, horizon)
    target = GuideTarget(actions=torch.ones(horizon, dim), weights=weights)
    x = torch.randn(1, horizon, dim)

    def v_fn(x_in):
        return 0.5 * x_in  # dx0/dx = 1 - 0.5 t > 0, as the guide assumes (J ~ I)

    x0 = x - t * v_fn(x)
    x0_guided = x - t * guide.correct_velocity(x, v_fn, t, target)
    assert torch.all((x0_guided[0, :3] - 1).abs() < (x0[0, :3] - 1).abs())
    assert torch.allclose(x0_guided[0, 3:], x0[0, 3:])


# -- engine with a fake policy ---------------------------------------------------------------------


class FakeAdapter:
    """Returns ``k_true + i`` for row ``i``, so a correctly aligned action equals its tick."""

    supports_guidance = False
    n_obs_steps = 1
    horizon_offset = 0
    accel = AccelConfig()
    device = torch.device("cpu")

    def __init__(self, length: int, latency_s: float) -> None:
        self.length = length
        self.latency_s = latency_s
        self.calls = 0

    def setup(self):
        pass

    def reset(self):
        pass

    def predict(self, frames, guide=None, target=None):
        self.calls += 1
        time.sleep(self.latency_s)
        k = frames[-1]["k_true"]
        return (k + np.arange(self.length, dtype=np.float64))[:, None].repeat(2, axis=1)


def run_ticks(engine: AsyncChunkEngine, fps: float, n: int) -> list[float]:
    out = []
    t0 = time.perf_counter()
    for k in range(n):
        while time.perf_counter() < t0 + k / fps:
            time.sleep(0.0005)
        a = engine.get_action({"k_true": k})
        out.append(float(a[0]))
    return out


def test_engine_aligns_actions_to_ticks_and_never_holds():
    fps = 50.0  # 20 ms ticks, 30 ms inference -> 2 stale rows per chunk
    adapter = FakeAdapter(length=20, latency_s=0.03)
    engine = AsyncChunkEngine(adapter, RemainingTrigger(6), SwitchBlender(), None, fps=fps)
    engine.reset()
    engine.start()
    try:
        got = run_ticks(engine, fps, 60)
    finally:
        engine.stop()
    # Tick 0 waited for the first chunk; every later action belongs to its own tick.
    mismatched = [k for k, v in enumerate(got) if v != k]
    assert len(mismatched) <= 2, mismatched
    ticks = [r["tick"] for r in engine._tick_rows]
    assert len(set(ticks)) == len(ticks), "a tick was counted twice"
    assert adapter.calls >= 3
    assert sum(r["held"] for r in engine._tick_rows) == 0


def test_engine_holds_when_inference_is_too_slow_and_reset_clears():
    fps = 50.0
    adapter = FakeAdapter(length=6, latency_s=0.15)  # 7.5 ticks of latency, 6-action chunks
    engine = AsyncChunkEngine(adapter, RemainingTrigger(2), SwitchBlender(), None, fps=fps)
    engine.reset()
    engine.start()
    try:
        run_ticks(engine, fps, 40)
        held = sum(r["held"] for r in engine._tick_rows)
        assert held > 0
        engine.reset()
        assert engine._blender.newest is None and not engine._history
    finally:
        engine.stop()


def test_guide_requires_capable_adapter():
    with pytest.raises(ValueError):
        AsyncChunkEngine(FakeAdapter(4, 0.0), RemainingTrigger(1), SwitchBlender(), InpaintGuide("exp"), fps=30)


# -- Flow Matching Policy adapter ------------------------------------------------------------------


def make_flow_matching_adapter(**config_kwargs):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.flow_matching import FlowMatchingConfig, FlowMatchingPolicy
    from lerobot.policies.flow_matching.processor_flow_matching import make_flow_matching_pre_post_processors
    from lerobot.rollout.inference.async_chunk.adapters import FlowMatchingAdapter, make_adapter

    identity = {k: NormalizationMode.IDENTITY for k in ("VISUAL", "STATE", "ACTION")}
    config = FlowMatchingConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        horizon=16,
        n_action_steps=8,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        pretrained_backbone_weights=None,
        normalization_mapping=identity,
        device="cpu",
        **config_kwargs,
    )
    policy = FlowMatchingPolicy(config).eval()
    pre, post = make_flow_matching_pre_post_processors(config, dataset_stats=None)
    names = ["a", "b", "c"]
    adapter = make_adapter(
        policy,
        preprocessor=pre,
        postprocessor=post,
        dataset_features={"action": {"names": names}},
        ordered_action_keys=names,
        task="",
        robot_type="",
        device="cpu",
        accel=AccelConfig(),
    )
    assert isinstance(adapter, FlowMatchingAdapter) and adapter.supports_guidance
    adapter.setup()
    frames = [
        {
            "observation.state": np.full(3, 0.1 * i, dtype=np.float32),
            "observation.images.front": np.full((64, 64, 3), 40 * i, dtype=np.uint8),
        }
        for i in range(2)
    ]
    return adapter, frames


def test_flow_matching_adapter_matches_predict_action_chunk():
    adapter, frames = make_flow_matching_adapter()
    assert adapter.n_obs_steps == 2 and adapter.horizon_offset == 1 and adapter.horizon == 16
    with torch.no_grad():
        torch.manual_seed(5)
        chunk = adapter.predict(frames)
        torch.manual_seed(5)
        batch = adapter._batch(frames)
        adapter.policy.reset()
        ref = adapter.policy.predict_action_chunk(batch)[0].numpy()
    assert chunk.shape == (8, 3)
    np.testing.assert_allclose(chunk, ref, rtol=1e-6, atol=1e-6)


def test_flow_matching_adapter_inpaint_freezes_prefix():
    adapter, frames = make_flow_matching_adapter(num_inference_steps=4)
    guide = InpaintGuide("exp")
    horizon = adapter.horizon
    weights = guide.prefix_weights(4, 10, horizon)
    target = GuideTarget(actions=torch.full((horizon, 3), 0.3), weights=weights)
    with torch.no_grad():
        chunk = adapter.predict(frames, guide=guide, target=target)
    # Sampler row j is chunk row j - horizon_offset: rows 1..3 of the horizon are frozen to the target.
    np.testing.assert_allclose(chunk[:3], 0.3, atol=1e-5)
    assert not np.allclose(chunk[-3:], 0.3, atol=1e-3)


def make_diffusion_adapter(accel: AccelConfig, device: str = "cuda", **config_kwargs):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy, make_diffusion_pre_post_processors
    from lerobot.rollout.inference.async_chunk.adapters import DiffusionAdapter, make_adapter

    identity = {k: NormalizationMode.IDENTITY for k in ("VISUAL", "STATE", "ACTION")}
    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        horizon=16,
        n_action_steps=8,
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        pretrained_backbone_weights=None,
        normalization_mapping=identity,
        num_inference_steps=4,
        device=device,
        use_amp=True,
        **config_kwargs,
    )
    torch.manual_seed(0)
    policy = DiffusionPolicy(config).to(device).eval()
    pre, post = make_diffusion_pre_post_processors(config, dataset_stats=None)
    names = ["a", "b", "c"]
    adapter = make_adapter(
        policy,
        preprocessor=pre,
        postprocessor=post,
        dataset_features={"action": {"names": names}},
        ordered_action_keys=names,
        task="",
        robot_type="",
        device=device,
        accel=accel,
    )
    assert isinstance(adapter, DiffusionAdapter)
    adapter.setup()
    frames = [
        {
            "observation.state": np.full(3, 0.1 * i, dtype=np.float32),
            "observation.images.front": np.full((64, 64, 3), 40 * i, dtype=np.uint8),
        }
        for i in range(2)
    ]
    return adapter, frames


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_diffusion_unet_cuda_graph_matches_eager():
    from lerobot.rollout.inference.async_chunk.adapters import _UNetGraph

    fast = dict(half_weights=True, channels_last=True, unet_cudnn=False)
    eager, frames = make_diffusion_adapter(AccelConfig(**fast))
    graphed, _ = make_diffusion_adapter(AccelConfig(**fast, cuda_graph=True))
    assert isinstance(graphed.policy.diffusion.unet.forward, _UNetGraph)
    assert graphed.policy.diffusion.unet.forward.key is not None  # captured during the warm-up
    graph = graphed.policy.diffusion.unet.forward
    graphed.setup()  # the engine calls setup again on every start: the U-Net must not be wrapped twice
    assert graphed.policy.diffusion.unet.forward is graph
    guide = InpaintGuide("exp")
    weights = guide.prefix_weights(4, 10, 16).to("cuda")
    target = GuideTarget(actions=torch.full((16, 3), 0.3, device="cuda"), weights=weights)
    for kwargs in ({}, {"guide": guide, "target": target}):  # predict_action_chunk and the guided sampler
        for _ in range(2):  # a replay after the first must still match (static buffers are refreshed)
            torch.manual_seed(7)
            ref = eager.predict(frames, **kwargs)
            torch.manual_seed(7)
            out = graphed.predict(frames, **kwargs)
            np.testing.assert_array_equal(out, ref)
            frames = frames[::-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_diffusion_unet_cuda_graph_runs_eagerly_with_grad():
    adapter, frames = make_diffusion_adapter(AccelConfig(cuda_graph=True))
    unet = adapter.policy.diffusion.unet
    x = torch.randn(1, 16, 3, device="cuda", requires_grad=True)
    t = torch.full((1,), 3, dtype=torch.long, device="cuda")
    gc = torch.randn(1, unet.forward.global_cond.shape[-1], device="cuda")
    with torch.autocast("cuda", cache_enabled=False):
        y = unet(x, t, global_cond=gc)
    y.float().sum().backward()  # the rtc guide differentiates through the U-Net: no graph there
    assert x.grad is not None and torch.isfinite(x.grad).all()
