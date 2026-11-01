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

"""How a newly arrived chunk is joined with what the robot is already executing (local change, 2026-10-22).

A blender receives finished chunks (:meth:`Blender.add`) and answers, for each control tick, which
action to send (:meth:`Blender.action`). It only post-processes finished chunks; steering the
sampling itself towards the previous chunk is a guide's job (``guides.py``).

To add a blender, subclass :class:`Blender`, add a config registered with
``BlendConfig.register_subclass`` and return the blender from ``build``.

The ``latest_only`` / ``weighted_average`` / ``average`` / ``conservative`` blenders reproduce the
aggregate functions of LeRobot's gRPC async inference (``async_inference/configs.py``,
``robot_client._aggregate_action_queues``) under the same names, so results can be compared with it.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import draccus
import numpy as np

from .timeline import Chunk


class Blender(abc.ABC):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def reset(self) -> None:
        self.chunks = []

    @property
    def newest(self) -> Chunk | None:
        return self.chunks[-1] if self.chunks else None

    def add(self, chunk: Chunk, k_next: int) -> None:
        """Take in a finished chunk. ``k_next`` is the first tick whose action has not been sent yet."""
        self.chunks.append(chunk)
        # Keep only the chunks that still cover a future tick (and always the newest).
        self.chunks = [c for c in self.chunks[:-1] if c.k_end > k_next] + [chunk]

    @abc.abstractmethod
    def action(self, k: int) -> np.ndarray | None:
        """Action for tick ``k``, or ``None`` if nothing covers it (the engine then holds the last action)."""

    def plan(self, k: int) -> np.ndarray | None:
        """What the robot is expected to execute at future tick ``k`` (used by guides). Defaults to :meth:`action`."""
        return self.action(k)


@dataclass
class BlendConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self) -> Blender: ...


class SwitchBlender(Blender):
    """Use the newest chunk from the tick it arrives; its stale head (the inference delay) is skipped."""

    def action(self, k: int) -> np.ndarray | None:
        for c in reversed(self.chunks):
            if c.covers(k):
                return c.at(k)
        return None


@BlendConfig.register_subclass("switch")
@dataclass
class SwitchBlendConfig(BlendConfig):
    def build(self) -> Blender:
        return SwitchBlender()


class AggregateBlender(Blender):
    """LeRobot async inference's queue merge: on arrival, each future tick becomes ``fn(old, new)``.

    ``old`` is whatever the queue already held for that tick (itself an earlier merge), so the
    result depends on the arrival order, exactly as in ``robot_client._aggregate_action_queues``.
    As there, the queue is rebuilt from the new chunk alone: ticks already sent are dropped and
    old entries the new chunk does not cover are discarded.
    """

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        super().__init__()
        self.fn = fn
        self.queue: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        super().reset()
        self.queue = {}

    def add(self, chunk: Chunk, k_next: int) -> None:
        super().add(chunk, k_next)
        merged: dict[int, np.ndarray] = {}
        for i, new in enumerate(chunk.actions):
            k = chunk.k_obs + i
            if k < k_next:
                continue
            old = self.queue.get(k)
            merged[k] = new if old is None else self.fn(old, new)
        self.queue = merged

    def action(self, k: int) -> np.ndarray | None:
        return self.queue.get(k)


# Same coefficients as LeRobot async_inference/configs.py AGGREGATE_FUNCTIONS.
_AGGREGATE_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@BlendConfig.register_subclass("weighted_average")
@dataclass
class WeightedAverageBlendConfig(BlendConfig):
    def build(self) -> Blender:
        return AggregateBlender(_AGGREGATE_FUNCTIONS["weighted_average"])


@BlendConfig.register_subclass("latest_only")
@dataclass
class LatestOnlyBlendConfig(BlendConfig):
    def build(self) -> Blender:
        return AggregateBlender(_AGGREGATE_FUNCTIONS["latest_only"])


@BlendConfig.register_subclass("average")
@dataclass
class AverageBlendConfig(BlendConfig):
    def build(self) -> Blender:
        return AggregateBlender(_AGGREGATE_FUNCTIONS["average"])


@BlendConfig.register_subclass("conservative")
@dataclass
class ConservativeBlendConfig(BlendConfig):
    def build(self) -> Blender:
        return AggregateBlender(_AGGREGATE_FUNCTIONS["conservative"])


class CrossfadeBlender(Blender):
    """Move linearly from what was being executed to the new chunk over ``ticks`` ticks.

    Chunk ``n`` fades in from tick ``k_ready``: ``out_n(k) = (1 - w) * out_{n-1}(k) + w * chunk_n(k)``
    with ``w = (k - k_ready + 1) / ticks`` clipped to [0, 1]. Where the older output no longer covers
    ``k``, the new chunk is used as is.
    """

    def __init__(self, ticks: int) -> None:
        super().__init__()
        self.ticks = max(1, ticks)

    def _out(self, n: int, k: int) -> np.ndarray | None:
        c = self.chunks[n]
        new = c.at(k) if c.covers(k) else None
        if n == 0:
            return new
        w = min(1.0, max(0.0, (k - c.k_ready + 1) / self.ticks))
        if new is None or w >= 1.0:
            return new if new is not None else self._out(n - 1, k)
        old = self._out(n - 1, k)
        return new if old is None else (1.0 - w) * old + w * new

    def action(self, k: int) -> np.ndarray | None:
        return self._out(len(self.chunks) - 1, k) if self.chunks else None


@BlendConfig.register_subclass("crossfade")
@dataclass
class CrossfadeBlendConfig(BlendConfig):
    # Ticks over which a new chunk takes over (1 = switch at once).
    ticks: int = 5

    def build(self) -> Blender:
        return CrossfadeBlender(self.ticks)


class EnsembleBlender(Blender):
    """ACT's temporal ensemble: average every chunk covering the tick, weight ``exp(-coeff * i)``, ``i = 0`` oldest."""

    def __init__(self, coeff: float, max_chunks: int) -> None:
        super().__init__()
        self.coeff = coeff
        self.max_chunks = max_chunks

    def add(self, chunk: Chunk, k_next: int) -> None:
        super().add(chunk, k_next)
        self.chunks = self.chunks[-self.max_chunks :]

    def action(self, k: int) -> np.ndarray | None:
        covering = [c.at(k) for c in self.chunks if c.covers(k)]
        if not covering:
            return None
        w = np.array([math.exp(-self.coeff * i) for i in range(len(covering))])
        return np.tensordot(w / w.sum(), np.stack(covering), axes=1)


@BlendConfig.register_subclass("ensemble")
@dataclass
class EnsembleBlendConfig(BlendConfig):
    # ACT's temporal_ensemble_coeff (positive = favour older predictions).
    coeff: float = 0.01
    max_chunks: int = 8

    def build(self) -> Blender:
        return EnsembleBlender(self.coeff, self.max_chunks)


class SmoothBlender(SwitchBlender):
    """``switch`` on chunks smoothed along time with a centred moving average of ``window`` rows.

    A chunk holds the planned future, so the average can look ahead and adds no lag (a causal filter
    on the sent actions would). Edges repeat the first / last row. Meant against tick-to-tick
    jitter in sampled chunks (residual sampling noise), not against jumps between chunks.
    """

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = max(1, window | 1)  # odd

    def add(self, chunk: Chunk, k_next: int) -> None:
        if self.window > 1 and len(chunk.actions) > 1:
            half = self.window // 2
            padded = np.pad(chunk.actions, ((half, half), (0, 0)), mode="edge")
            kernel = np.ones(self.window) / self.window
            smoothed = np.stack(
                [np.convolve(padded[:, j], kernel, mode="valid") for j in range(chunk.actions.shape[1])], axis=1
            )
            chunk = replace(chunk, actions=smoothed)
        super().add(chunk, k_next)


@BlendConfig.register_subclass("smooth")
@dataclass
class SmoothBlendConfig(BlendConfig):
    # Rows of the centred moving average (odd; 1 = plain switch).
    window: int = 5

    def build(self) -> Blender:
        return SmoothBlender(self.window)
