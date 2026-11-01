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

"""When to start computing the next chunk (local change, 2026-10-22).

The worker asks its trigger right after each new observation arrives, and never runs two
inferences at once. To add a trigger, subclass :class:`Trigger`, add a config registered with
``TriggerConfig.register_subclass`` and return the trigger from ``build``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import draccus

from .timeline import Chunk


class Trigger(abc.ABC):
    @abc.abstractmethod
    def should_start(self, k_now: int, newest: Chunk | None, last_start: int | None) -> bool:
        """``k_now``: tick of the newest observation. ``last_start``: ``k_now`` of the previous start."""


@dataclass
class TriggerConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self) -> Trigger: ...


class RemainingTrigger(Trigger):
    """Start when the newest chunk has at most ``k`` actions left (LeRobot async's ``chunk_size_threshold``)."""

    def __init__(self, k: int) -> None:
        self.k = k

    def should_start(self, k_now: int, newest: Chunk | None, last_start: int | None) -> bool:
        return newest is None or newest.remaining(k_now) <= self.k


@TriggerConfig.register_subclass("remaining")
@dataclass
class RemainingTriggerConfig(TriggerConfig):
    # Actions left in the newest chunk at which the next inference starts. Should exceed the
    # inference delay in ticks, or the robot holds its last action until the chunk arrives.
    k: int = 10

    def build(self) -> Trigger:
        return RemainingTrigger(self.k)


class PeriodTrigger(Trigger):
    """Start every ``n`` ticks: re-plan at a fixed rate, independent of the chunk length."""

    def __init__(self, n: int) -> None:
        self.n = n

    def should_start(self, k_now: int, newest: Chunk | None, last_start: int | None) -> bool:
        return newest is None or last_start is None or k_now - last_start >= self.n


@TriggerConfig.register_subclass("period")
@dataclass
class PeriodTriggerConfig(TriggerConfig):
    n: int = 8

    def build(self) -> Trigger:
        return PeriodTrigger(self.n)


class ContinuousTrigger(Trigger):
    """Start again as soon as the previous inference finished."""

    def should_start(self, k_now: int, newest: Chunk | None, last_start: int | None) -> bool:
        return True


@TriggerConfig.register_subclass("continuous")
@dataclass
class ContinuousTriggerConfig(TriggerConfig):
    def build(self) -> Trigger:
        return ContinuousTrigger()
