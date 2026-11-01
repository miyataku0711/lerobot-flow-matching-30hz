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

"""Async-chunk inference (local change, 2026-10-22): see ``engine.py``.

Importing this package registers ``--inference.type=async_chunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from ..factory import InferenceEngineConfig
from .adapters import (
    AccelConfig,
    ChunkPolicyAdapter,
    DiffusionAdapter,
    FlowMatchingAdapter,
    GenericAdapter,
    make_adapter,
)
from .blenders import Blender, BlendConfig, SwitchBlendConfig
from .engine import AsyncChunkEngine
from .guides import Guide, GuideConfig, GuideTarget, NoGuideConfig
from .timeline import Chunk, TickClock
from .triggers import RemainingTriggerConfig, Trigger, TriggerConfig


@InferenceEngineConfig.register_subclass("async_chunk")
@dataclass
class AsyncChunkInferenceConfig(InferenceEngineConfig):
    """Compute chunks in a background thread; trigger, blender, guide and speed-ups are pluggable."""

    trigger: TriggerConfig = field(default_factory=RemainingTriggerConfig)
    blend: BlendConfig = field(default_factory=SwitchBlendConfig)
    guide: GuideConfig = field(default_factory=NoGuideConfig)
    accel: AccelConfig = field(default_factory=AccelConfig)
    # Per-chunk and per-tick CSV logs are written here when the engine stops (None = no logs).
    log_dir: str | None = None
    # How long get_action waits for the first chunk of an episode.
    first_chunk_timeout_s: float = 10.0


def build_async_chunk_engine(
    config: AsyncChunkInferenceConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    robot_type: str,
    fps: float,
    device: str | None,
    shutdown_event: Event | None = None,
) -> AsyncChunkEngine:
    adapter = make_adapter(
        policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task,
        robot_type=robot_type,
        device=device,
        accel=config.accel,
    )
    return AsyncChunkEngine(
        adapter=adapter,
        trigger=config.trigger.build(),
        blender=config.blend.build(),
        guide=config.guide.build(),
        fps=fps,
        first_chunk_timeout_s=config.first_chunk_timeout_s,
        log_dir=config.log_dir,
        shutdown_event=shutdown_event,
    )


__all__ = [
    "AccelConfig",
    "AsyncChunkEngine",
    "AsyncChunkInferenceConfig",
    "Blender",
    "BlendConfig",
    "Chunk",
    "ChunkPolicyAdapter",
    "DiffusionAdapter",
    "FlowMatchingAdapter",
    "GenericAdapter",
    "Guide",
    "GuideConfig",
    "GuideTarget",
    "TickClock",
    "Trigger",
    "TriggerConfig",
    "build_async_chunk_engine",
    "make_adapter",
]
