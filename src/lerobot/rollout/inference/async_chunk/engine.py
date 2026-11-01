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

"""Async-chunk inference engine: compute the next chunk in a background thread (local change, 2026-10-22).

The control thread never touches the GPU. Each ``get_action`` stamps the observation frame with
its control tick, appends it to a short history, wakes the worker and asks the blender for the
action of the current tick. The worker asks the trigger whether to start, has the adapter predict a
chunk from the history (optionally steered by a guide towards what will already have been executed),
and hands it to the blender. If no chunk covers the current tick the last action is held.

The four parts are independent and selected on the CLI::

    --inference.type=async_chunk
    --inference.trigger.type=remaining|period|continuous    (triggers.py)
    --inference.blend.type=switch|crossfade|ensemble|weighted_average|latest_only|average|conservative
    --inference.guide.type=none|inpaint|rtc                (guides.py, Diffusion, FlowMatching, SmolVLA)
    --inference.accel.half_weights=true ...                (adapters.AccelConfig)
"""

from __future__ import annotations

import csv
import logging
import math
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch

from ..base import InferenceEngine
from .adapters import ChunkPolicyAdapter
from .blenders import Blender
from .guides import Guide, GuideTarget
from .timeline import Chunk, TickClock
from .triggers import Trigger

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_S = 3.0
_IDLE_WAIT_S = 0.05
_MAX_CONSECUTIVE_ERRORS = 10


class AsyncChunkEngine(InferenceEngine):
    def __init__(
        self,
        adapter: ChunkPolicyAdapter,
        trigger: Trigger,
        blender: Blender,
        guide: Guide | None,
        fps: float,
        first_chunk_timeout_s: float = 10.0,
        log_dir: str | None = None,
        shutdown_event: Event | None = None,
    ) -> None:
        if guide is not None and not adapter.supports_guidance:
            raise ValueError(f"guide '{type(guide).__name__}' needs an adapter that exposes its sampler")
        self._adapter = adapter
        self._trigger = trigger
        self._blender = blender
        self._guide = guide
        self._clock = TickClock(fps)
        self._first_chunk_timeout_s = first_chunk_timeout_s
        self._log_dir = Path(log_dir).expanduser() if log_dir else None
        self._global_shutdown = shutdown_event

        self._lock = Lock()
        self._obs_event = Event()
        self._chunk_event = Event()
        self._active = Event()
        self._active.set()
        self._stop = Event()
        self._error = Event()
        self._thread: Thread | None = None

        self._history: deque[tuple[int, float, dict]] = deque(maxlen=max(1, adapter.n_obs_steps))
        self._sent: dict[int, np.ndarray] = {}  # tick -> action actually returned (for guides)
        self._k_last_sent = -1
        self._last_action: np.ndarray | None = None
        self._last_start: int | None = None
        self._next_id = 0
        self._epoch = 0
        self._latencies: deque[float] = deque(maxlen=20)
        self._chunk_rows: list[dict] = []
        self._tick_rows: list[dict] = []
        self._setup_done = Event()

    # -- lifecycle -------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._setup_done.is_set()

    @property
    def failed(self) -> bool:
        return self._error.is_set()

    def start(self) -> None:
        self._stop.clear()
        self._thread = Thread(target=self._worker, daemon=True, name="AsyncChunkInference")
        self._thread.start()
        self._setup_done.wait(timeout=60.0)
        logger.info("AsyncChunkEngine started (%s)", self.describe())

    def stop(self) -> None:
        self._stop.set()
        self._obs_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("AsyncChunkInference thread did not join within %.1fs", _JOIN_TIMEOUT_S)
        self._thread = None
        self._flush_logs()

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    def reset(self) -> None:
        with self._lock:
            self._epoch += 1  # an inference in flight belongs to the old episode and is dropped
            self._clock.reset()
            self._history.clear()
            self._blender.reset()
            self._sent = {}
            self._k_last_sent = -1
            self._last_action = None
            self._last_start = None
            self._chunk_event.clear()
        self._adapter.reset()

    def describe(self) -> str:
        guide = type(self._guide).__name__ if self._guide else "none"
        return (
            f"adapter={type(self._adapter).__name__} trigger={type(self._trigger).__name__} "
            f"blender={type(self._blender).__name__} guide={guide} accel={self._adapter.accel}"
        )

    # -- control thread ----------------------------------------------------------------------------
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None
        t = time.perf_counter()
        with self._lock:
            k = self._clock.tick(t)
            if not self._history or self._history[-1][0] != k:
                self._history.append((k, t, obs_frame))
            first = self._blender.newest is None
        self._obs_event.set()
        if first:
            # Nothing to execute yet (start of an episode): wait for the first chunk, like the sync
            # engine's first tick. This also absorbs CUDA/cuDNN initialisation in the warm-up call.
            if not self._chunk_event.wait(timeout=self._first_chunk_timeout_s):
                logger.warning("No action chunk within %.1fs", self._first_chunk_timeout_s)
        with self._lock:
            action = self._blender.action(k)
            newest = self._blender.newest
            held = action is None
            if held:
                action = self._last_action
            if action is None:
                return None
            self._last_action = action
            self._sent[k] = action
            self._k_last_sent = k
            for old in [x for x in self._sent if x < k - 256]:
                del self._sent[old]
            self._tick_rows.append(
                {
                    "epoch": self._epoch,
                    "tick": k,
                    "t": t,
                    "chunk_id": newest.id if newest else -1,
                    "age": k - newest.k_obs if newest else -1,
                    "held": int(held),
                    **{f"a{i}": float(v) for i, v in enumerate(action)},
                }
            )
        return torch.as_tensor(action, dtype=torch.float32)

    # -- inference thread --------------------------------------------------------------------------
    def _worker(self) -> None:
        try:
            self._adapter.setup()
        except Exception:
            logger.error("Async-chunk adapter setup failed:\n%s", traceback.format_exc())
            self._fail()
            return
        finally:
            self._setup_done.set()
        errors = 0
        while not self._stop.is_set():
            if not self._obs_event.wait(timeout=_IDLE_WAIT_S):
                continue
            self._obs_event.clear()
            if self._stop.is_set() or not self._active.is_set():
                continue
            with self._lock:
                if not self._history:
                    continue
                k_obs, t_obs, _ = self._history[-1]
                if not self._trigger.should_start(k_obs, self._blender.newest, self._last_start):
                    continue
                self._last_start = k_obs
                frames = [f for _, _, f in self._history]
                epoch = self._epoch
                target = self._guide_target(k_obs) if self._guide is not None else None
            try:
                t_start = time.perf_counter()
                actions = self._adapter.predict(frames, self._guide, target)
                t_ready = time.perf_counter()
                errors = 0
            except Exception as e:
                errors += 1
                logger.error("Async-chunk inference error (%d/%d): %s", errors, _MAX_CONSECUTIVE_ERRORS, e)
                logger.debug(traceback.format_exc())
                if errors >= _MAX_CONSECUTIVE_ERRORS:
                    self._fail()
                    return
                continue
            self._latencies.append(t_ready - t_start)
            with self._lock:
                if epoch != self._epoch:
                    continue
                k_next = self._k_last_sent + 1
                chunk = Chunk(
                    id=self._next_id,
                    k_obs=k_obs,
                    actions=actions,
                    t_obs=t_obs,
                    t_start=t_start,
                    t_ready=t_ready,
                    k_ready=k_next,
                )
                self._next_id += 1
                self._blender.add(chunk, k_next)
                self._chunk_rows.append(
                    {
                        "epoch": epoch,
                        "chunk_id": chunk.id,
                        "k_obs": k_obs,
                        "k_ready": k_next,
                        "delay_ticks": chunk.delay,
                        "latency_ms": (t_ready - t_start) * 1e3,
                        "obs_age_at_start_ms": (t_start - t_obs) * 1e3,
                        "length": len(actions),
                        "guided": int(target is not None),
                    }
                )
            self._chunk_event.set()

    def _fail(self) -> None:
        self._error.set()
        self._chunk_event.set()
        if self._global_shutdown is not None:
            self._global_shutdown.set()

    def _expected_delay(self) -> int:
        """Ticks the next chunk will take, from the slowest recent inference (as LeRobot RTC does)."""
        if not self._latencies:
            return 0
        return math.ceil(max(self._latencies) / self._clock.dt)

    def _guide_target(self, k_obs: int) -> GuideTarget | None:
        """Rows of the new chunk tied to what was / will be executed (called with the lock held)."""
        if self._blender.newest is None:
            return None
        adapter, horizon = self._adapter, self._adapter.horizon
        k0 = k_obs - adapter.horizon_offset
        rows, actions = [], []
        for j in range(horizon):
            k = k0 + j
            a = self._sent.get(k)
            if a is None:
                a = self._blender.plan(k)
            if a is None:
                break
            rows.append(j)
            actions.append(a)
        if not rows:
            return None
        # Rows before k_obs were sent; k_obs .. k_obs + d - 1 will be sent before the chunk arrives.
        frozen = min(len(rows), adapter.horizon_offset + max(1, self._expected_delay()))
        weights = self._guide.prefix_weights(frozen, len(rows), horizon)
        target = torch.zeros(horizon, len(actions[0]), device=adapter.device)
        target[: len(rows)] = adapter.to_model_space(np.stack(actions))
        return GuideTarget(actions=target, weights=weights.to(adapter.device))

    # -- logs --------------------------------------------------------------------------------------
    def _flush_logs(self) -> None:
        if self._log_dir is None or not (self._chunk_rows or self._tick_rows):
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, rows in (("chunks", self._chunk_rows), ("ticks", self._tick_rows)):
            if not rows:
                continue
            path = self._log_dir / f"async_{name}_{stamp}.csv"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            logger.info("Saved %d async-chunk %s rows to %s", len(rows), name, path)
        self._chunk_rows, self._tick_rows = [], []
