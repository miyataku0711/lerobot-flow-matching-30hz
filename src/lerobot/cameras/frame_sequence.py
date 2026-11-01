#!/usr/bin/env python

# Local addition (2026-09-28)

"""Pick camera frames in sequence instead of "whatever is newest".

A control loop and a camera run on separate clocks at the same nominal rate, so
the moment the loop samples and the moment a new frame lands drift into a fixed
relation (phase). Taking the newest frame works when that phase is far from a
frame arrival; when it sits near one, arrival jitter (±2.7 ms on an EMEET C960,
±1.5 ms on a RealSense) flips every tick between "already here" and "not yet",
so the same frame is used twice and the next one is skipped, over and over.
Measured on the SO-101 rig: up to 393 of 600 ticks in one episode.

:class:`FrameSequenceMixin` keeps the last few frames with a sequence number and
hands out *the frame after the one used last time*, waiting briefly for it if it
is a hair late. The frame chosen no longer depends on which side of the arrival
the tick fell, so the jitter stops mattering. The cost is that the image can be
one frame older than the newest, but its age stays constant.

Set ``LEROBOT_FRAME_SELECT=latest`` to go back to newest-frame selection.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Condition
from typing import Any

FRAME_SELECT_ENV = "LEROBOT_FRAME_SELECT"
# A chosen frame older than this (two periods at 30 fps) is treated as stale: the reader waits up to
# RESTART_WAIT_S for the next frame and restarts the sequence from the newest one.
STALE_RESTART_S = 0.066
RESTART_WAIT_S = 0.2  # enough to drain a few buffered frames after a stall


def sequenced_selection_enabled() -> bool:
    return os.environ.get(FRAME_SELECT_ENV, "sequenced") != "latest"


@dataclass
class SequencedFrame:
    seq: int
    color: Any
    depth: Any
    timing: dict[str, float] = field(default_factory=dict)


def _frame_age(frame: SequencedFrame) -> float:
    """Seconds since the frame was shot (sensor/driver time), or since it was read if that is unknown."""
    t = frame.timing.get("device", frame.timing.get("capture"))
    return time.perf_counter() - t if t is not None else 0.0


class FrameSequenceMixin:
    """Ring buffer of recent frames plus in-sequence selection.

    The camera calls :meth:`_init_frame_sequence` once (after creating
    ``self.frame_lock``), :meth:`_push_frame` from its read thread while holding
    ``self.frame_lock``, and :meth:`_reset_frame_sequence` when the stream stops.
    """

    FRAME_HISTORY = 4

    def _init_frame_sequence(self) -> None:
        self._frames: deque[SequencedFrame] = deque(maxlen=self.FRAME_HISTORY)
        self._frame_seq = 0
        self._frame_cond = Condition(self.frame_lock)  # type: ignore[attr-defined]

    def _push_frame(self, color: Any, depth: Any, timing: dict[str, float]) -> None:
        """Record a new frame. Call with ``self.frame_lock`` held."""
        self._frame_seq += 1
        self._frames.append(SequencedFrame(self._frame_seq, color, depth, timing))
        self._frame_cond.notify_all()

    def _reset_frame_sequence(self) -> None:
        """Drop the history. Call with ``self.frame_lock`` held."""
        self._frames.clear()

    def _frame(self, seq: int) -> SequencedFrame | None:
        if not self._frames:
            return None
        i = seq - self._frames[0].seq
        return self._frames[i] if 0 <= i < len(self._frames) else None

    def read_next_frame(
        self,
        last_seq: int | None,
        wait_s: float = 0.004,
        catchup_s: float = 0.004,
        max_age_ms: float = 500,
    ) -> SequencedFrame:
        """Return the frame after ``last_seq`` (or the newest one to start with).

        - If frame ``last_seq + 1`` is here, return it, even when a newer one is too.
        - If it is not here yet, wait up to ``wait_s`` for it; if it still does not
          come, return ``last_seq`` again (the camera fell behind the loop).
        - If frame ``last_seq + 2`` arrived at least ``catchup_s`` ago, the loop has
          fallen a frame behind the camera: skip ahead instead of lagging forever.
        - If ``last_seq`` is ``None`` or no longer in the history, start again from
          the newest frame.
        """
        deadline = time.perf_counter() + wait_s
        with self._frame_cond:
            if not self._frames:
                # Nothing captured yet: wait like read_latest's first frame would.
                self._frame_cond.wait_for(lambda: bool(self._frames), timeout=max(wait_s, 1.0))
                if not self._frames:
                    raise RuntimeError(f"{self} has not captured any frames yet.")

            if last_seq is None or self._frame(last_seq) is None:
                chosen = self._frames[-1]
            else:
                now = time.perf_counter()
                while (ahead := self._frame(last_seq + 2)) is not None and ahead.timing.get(
                    "arrival", now
                ) <= now - catchup_s:
                    last_seq += 1
                nxt = last_seq + 1
                self._frame_cond.wait_for(
                    lambda: self._frame(nxt) is not None, timeout=max(deadline - time.perf_counter(), 0.0)
                )
                chosen = self._frame(nxt) or self._frame(last_seq) or self._frames[-1]

            # After a pause (e.g. an episode was just saved and the read threads were starved while
            # the next episode's encoders were being opened), the frame in sequence can be half a second
            # old. Wait briefly for a fresh frame and restart from the newest instead of starting the
            # episode on a stale image.
            # Age is taken from the sensor/driver timestamp when there is one: after a stall the driver's
            # buffers still hold frames shot before it, and the read thread hands those out with a fresh
            # host "capture" time.
            if _frame_age(chosen) > STALE_RESTART_S:
                self._frame_cond.wait_for(
                    lambda: _frame_age(self._frames[-1]) <= STALE_RESTART_S, timeout=RESTART_WAIT_S
                )
                chosen = self._frames[-1]
            capture = chosen.timing.get("capture")
            if capture is not None and (time.perf_counter() - capture) * 1e3 > max_age_ms:
                raise TimeoutError(
                    f"{self} frame {chosen.seq} is too old: "
                    f"{(time.perf_counter() - capture) * 1e3:.1f} ms (max allowed: {max_age_ms} ms)."
                )
            return chosen
