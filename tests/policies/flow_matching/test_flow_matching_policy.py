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

"""Tests for the Flow Matching policy.

To run tests locally:
    python -m pytest tests/policies/flow_matching/test_flow_matching.py -v
"""

import pytest
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import seeded_context, set_seed

STATE_DIM = 6
ACTION_DIM = 6
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
IMG = (3, 64, 64)


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(17)


def create_config(**kwargs) -> FlowMatchingConfig:
    kwargs = {"horizon": HORIZON, **kwargs}
    config = FlowMatchingConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=IMG),
            f"{OBS_IMAGES}.left": PolicyFeature(type=FeatureType.VISUAL, shape=IMG),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
        # Small and offline for fast tests.
        down_dims=(32, 64),
        diffusion_step_embed_dim=32,
        pretrained_backbone_weights=None,
        device="cpu",
        **kwargs,
    )
    config.validate_features()
    return config


def create_train_batch(batch_size: int = 2) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, N_OBS_STEPS, STATE_DIM),
        f"{OBS_IMAGES}.front": torch.rand(batch_size, N_OBS_STEPS, *IMG),
        f"{OBS_IMAGES}.left": torch.rand(batch_size, N_OBS_STEPS, *IMG),
        ACTION: torch.randn(batch_size, HORIZON, ACTION_DIM).clamp(-1, 1),
        "action_is_pad": torch.zeros(batch_size, HORIZON, dtype=torch.bool),
    }


def create_observation_batch(batch_size: int = 2) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        f"{OBS_IMAGES}.front": torch.rand(batch_size, *IMG),
        f"{OBS_IMAGES}.left": torch.rand(batch_size, *IMG),
    }


def test_flow_matching_registered_in_factory():
    assert make_policy_config("flow_matching", device="cpu").type == "flow_matching"
    assert get_policy_class("flow_matching") is FlowMatchingPolicy


@pytest.mark.parametrize("time_sampling", ["beta", "uniform"])
def test_flow_matching_forward_and_backward(time_sampling: str):
    policy = FlowMatchingPolicy(create_config(time_sampling=time_sampling))
    policy.train()
    loss, output_dict = policy.forward(create_train_batch())
    assert output_dict is None
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in policy.diffusion.unet.parameters() if p.requires_grad)


def test_flow_matching_loss_masks_padding():
    policy = FlowMatchingPolicy(create_config(do_mask_loss_for_padding=True))
    batch = create_train_batch()
    batch["action_is_pad"][:, -4:] = True
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


def test_flow_matching_select_action():
    policy = FlowMatchingPolicy(create_config())
    policy.eval()
    obs = create_observation_batch()
    with torch.no_grad():
        action = policy.select_action(obs)
        assert action.shape == (2, ACTION_DIM)
        # The rest of the chunk is served from the queue.
        assert len(policy._queues[ACTION]) == N_ACTION_STEPS - 1


def test_flow_matching_predict_action_chunk_is_deterministic_given_noise():
    policy = FlowMatchingPolicy(create_config())
    policy.eval()
    batch = create_train_batch()
    batch.pop(ACTION)
    noise = torch.randn(2, HORIZON, ACTION_DIM)
    with torch.no_grad():
        a = policy.predict_action_chunk(batch, noise=noise)
        b = policy.predict_action_chunk(batch, noise=noise)
    assert a.shape == (2, N_ACTION_STEPS, ACTION_DIM)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("num_steps", [1, 4])
def test_flow_matching_sampler_follows_the_straight_path(num_steps: int):
    """With the exact velocity of the straight path to `target`, Euler lands on `target` (t=1 is noise)."""
    policy = FlowMatchingPolicy(create_config(num_inference_steps=num_steps))
    model = policy.diffusion
    target = torch.rand(2, HORIZON, ACTION_DIM) * 1.6 - 0.8
    calls = []

    def exact_velocity(x_t, time, global_cond):
        calls.append(float(time[0]))
        return (x_t - target) / time[:, None, None]

    model.velocity = exact_velocity
    out = model.conditional_sample(2, global_cond=None, noise=torch.randn(2, HORIZON, ACTION_DIM))
    torch.testing.assert_close(out, target, rtol=1e-5, atol=1e-5)
    assert len(calls) == num_steps
    assert calls[0] == pytest.approx(1.0)


def test_flow_matching_loss_target_is_noise_minus_actions():
    """The loss is zero exactly when the network predicts `noise - actions` at `t * noise + (1 - t) * a`."""
    policy = FlowMatchingPolicy(create_config())
    model = policy.diffusion
    batch = create_train_batch()
    batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
    actions = batch[ACTION]

    def oracle(x_t, time, global_cond):
        t = time[:, None, None]
        # Recover the noise from the point on the path, then return the true velocity.
        noise = (x_t - (1 - t) * actions) / t
        return noise - actions

    model.velocity = oracle
    assert model.compute_loss(batch).item() == pytest.approx(0.0, abs=1e-8)


def test_flow_matching_clip_sample():
    policy = FlowMatchingPolicy(create_config(num_inference_steps=1, clip_sample_range=0.5))
    model = policy.diffusion
    model.velocity = lambda x_t, time, global_cond: torch.full_like(x_t, -10.0)
    out = model.conditional_sample(1, global_cond=None, noise=torch.zeros(1, HORIZON, ACTION_DIM))
    assert out.max().item() == pytest.approx(0.5)


def test_flow_matching_save_and_load(tmp_path):
    config = create_config()
    policy = FlowMatchingPolicy(config)
    policy.eval()
    policy.save_pretrained(tmp_path)
    loaded = FlowMatchingPolicy.from_pretrained(tmp_path)
    loaded.eval()
    assert loaded.config.type == "flow_matching"
    assert loaded.config.num_inference_steps == config.num_inference_steps

    batch = create_train_batch()
    batch.pop(ACTION)
    noise = torch.randn(2, HORIZON, ACTION_DIM)
    with torch.no_grad(), seeded_context(3):
        a = policy.predict_action_chunk(batch, noise=noise)
    with torch.no_grad(), seeded_context(3):
        b = loaded.predict_action_chunk(batch, noise=noise)
    torch.testing.assert_close(a, b)


def test_flow_matching_config_validation():
    with pytest.raises(ValueError, match="time_sampling"):
        create_config(time_sampling="logit_normal")
    with pytest.raises(ValueError, match="num_inference_steps"):
        create_config(num_inference_steps=0)
    with pytest.raises(ValueError, match="downsampling factor"):
        create_config(horizon=10)
