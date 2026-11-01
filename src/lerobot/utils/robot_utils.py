# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
import platform
import time


def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.005):
    """
    Wait for `seconds` with better precision than time.sleep alone at the expense of more CPU usage.

    Parameters:
      - seconds: duration to wait
      - spin_threshold: if remaining <= spin_threshold -> spin; otherwise sleep (seconds). Default 10ms
      - sleep_margin: when sleeping leave this much time before deadline to avoid oversleep. Default 5ms

    Note:
        The default parameters are chosen to prioritize timing accuracy over CPU usage for the common 30 FPS use case.
    """
    if seconds <= 0:
        return
    if spin_threshold < 0:
        raise ValueError(f"spin_threshold must be >= 0, got {spin_threshold}")
    if sleep_margin < 0:
        raise ValueError(f"sleep_margin must be >= 0, got {sleep_margin}")

    system = platform.system()
    # On macOS and Windows the scheduler / sleep granularity can make
    # short sleeps inaccurate. Instead of burning CPU for the whole
    # duration, sleep for most of the time and spin for the final few
    # milliseconds to achieve good accuracy with much lower CPU usage.
    if system in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            # If there's more than a couple milliseconds left, sleep most
            # of the remaining time and leave a small margin for the final spin.
            if remaining > spin_threshold:
                # Sleep but avoid sleeping past the end by leaving a small margin.
                time.sleep(max(remaining - sleep_margin, 0))
            else:
                # Final short spin to hit precise timing without long sleeps.
                pass
    else:
        # On Linux time.sleep is accurate enough for most uses
        time.sleep(seconds)


# Local change (2026-09-28): absolute-deadline pacing for control loops.
#
# The loops above sleep ``interval - dt`` relative to their own start, so every
# oversleep carries into the next period. On the DGX Spark a >=10 ms
# ``time.sleep`` wakes ~0.5 ms late (deep cpuidle exit), which ran the 30 Hz loops
# at ~29.65 Hz: the dataset clock (frame_index / fps) then lags real time by ~1%,
# and a 30 fps camera read by a slower loop skips one frame every ~3 s.


def sleep_until(deadline: float, coarse_margin: float = 0.002, spin_threshold: float = 0.0002) -> None:
    """Sleep until ``time.perf_counter()`` reaches ``deadline``.

    One long sleep stops ``coarse_margin`` short of the deadline (absorbing the
    deep-idle wake-up latency), a short sleep closes most of the rest, and only the
    last ``spin_threshold`` is spun. Spinning holds the GIL, so it is kept short to
    leave the camera and encoder threads alone.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > coarse_margin:
            time.sleep(remaining - coarse_margin)
        elif remaining > spin_threshold:
            time.sleep(remaining - spin_threshold)
        # else: spin


# Set to "relative" to get the old ``precise_sleep(interval - dt)`` pacing back, e.g.
# to record a before/after comparison with the same code and the same timing columns.
LOOP_PACING_ENV = "LEROBOT_LOOP_PACING"


class LoopClock:
    """Pace a loop on absolute deadlines ``t0 + k * interval``.

    Call :meth:`tick` at the top of each iteration and :meth:`sleep` at the end.
    A late iteration is made up within the next period; one more than a full
    period late re-anchors the schedule instead of running a burst of
    back-to-back iterations to catch up.

    With ``LEROBOT_LOOP_PACING=relative`` it falls back to the previous relative
    pacing; :meth:`tick` then returns the actual start, so no lateness is reported.
    """

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self._target: float | None = None
        self.relative = os.environ.get(LOOP_PACING_ENV, "absolute") == "relative"

    def tick(self) -> float:
        """Mark the start of an iteration and return the time it was scheduled to start."""
        now = time.perf_counter()
        if self.relative or self._target is None or now - self._target > self.interval_s:
            self._target = now
        return self._target

    def sleep(self) -> None:
        """Sleep until the scheduled start of the next iteration."""
        if self._target is None:
            return
        if self.relative:
            precise_sleep(max(self.interval_s - (time.perf_counter() - self._target), 0.0))
            return
        self._target += self.interval_s
        sleep_until(self._target)
