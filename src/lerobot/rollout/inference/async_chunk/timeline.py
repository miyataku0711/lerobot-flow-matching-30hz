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

"""Absolute tick indexing shared by the async-chunk engine and its plug-ins (local change, 2026-10-22).

Every action is addressed by the control tick it is meant for, not by its position in a queue.
A chunk predicted from the observation taken at tick ``k_obs`` holds the action for tick
``k_obs + i`` at row ``i``. Because ticks come from wall-clock time, a late or skipped control
period shifts nothing: the actions that went stale while the chunk was being computed are simply
never asked for. Triggers, blenders and guides only ever reason in ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TickClock:
    """Number the control ticks of period ``1 / fps`` the way ``LoopClock`` schedules them.

    ``LoopClock`` makes up a late iteration within the next period and only re-anchors its schedule
    when an iteration starts more than a full period late. Counting ticks by rounding the call time
    instead disagrees after a late iteration: on the real robot the first tick waited 50 ms for the
    first chunk, the second call came at 1.6 periods and was counted as tick 2, and the third (back on
    the schedule) as tick 2 again. So: a call less than a full period late is the next tick; a later
    call skips the ticks that went by and becomes the new anchor; a second call within half a period
    belongs to the same tick.
    """

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self._target: float | None = None
        self._k = 0

    def reset(self) -> None:
        self._target = None
        self._k = 0

    def tick(self, t: float) -> int:
        if self._target is None:
            self._target = t
            return self._k
        if t - self._target < 0.5 * self.dt:
            return self._k
        expected = self._target + self.dt
        if t - expected > self.dt:
            self._k += max(1, int((t - self._target) / self.dt))
            self._target = t
        else:
            self._k += 1
            self._target = expected
        return self._k


@dataclass
class Chunk:
    """One predicted action chunk, in robot action space and the rollout's action-key order.

    ``actions[i]`` is the action for tick ``k_obs + i``. ``k_ready`` is the first tick that can use it
    (the tick after the one that was current when the chunk arrived), so ``k_ready - k_obs`` is the
    inference delay in ticks.
    """

    id: int
    k_obs: int
    actions: np.ndarray  # (T, A)
    t_obs: float
    t_start: float
    t_ready: float
    k_ready: int

    @property
    def k_end(self) -> int:
        """One past the last tick this chunk covers."""
        return self.k_obs + len(self.actions)

    @property
    def delay(self) -> int:
        return self.k_ready - self.k_obs

    def covers(self, k: int) -> bool:
        return self.k_obs <= k < self.k_end

    def at(self, k: int) -> np.ndarray:
        return self.actions[k - self.k_obs]

    def remaining(self, k: int) -> int:
        """Number of actions left for ticks ``>= k``."""
        return max(0, self.k_end - k)
