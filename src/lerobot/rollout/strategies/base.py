# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from lerobot.utils.robot_utils import LoopClock
from lerobot.utils.timestamp_logger import TimestampLogger

from ..configs import BaseStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.

    When ``strategy.timestamp_log_dir`` is set, every control tick is timed and
    written to one parquet at the end of the run, so control-period jitter can be
    measured.  This strategy is the cleanest place to measure it: there is no
    dataset writing or video encoding competing with the control loop.
    """

    config: BaseStrategyConfig

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        self._prewarm_compiled_inference(ctx)

        control_interval = interpolator.get_control_interval(cfg.fps)
        loop_clock = LoopClock(control_interval)

        self._timestamp_logger = TimestampLogger() if self.config.timestamp_log_dir else None

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        try:
            while not ctx.runtime.shutdown_event.is_set():
                target_start = loop_clock.tick()
                loop_start = time.perf_counter()

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                obs = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                # Spans policy inference, interpolation and send_action.
                action_timestamp = time.perf_counter()
                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                loop_control_end_timestamp = time.perf_counter()

                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                self._record_tick(obs, loop_start, action_timestamp, loop_control_end_timestamp,
                                  target_start=target_start, action_sent=action_dict is not None)

                dt = time.perf_counter() - loop_start
                loop_clock.sleep()
                if control_interval - dt <= 0:
                    logger.warning(
                        f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                    )
        finally:
            # Also reached on Ctrl-C, so an interrupted run still keeps its timings.
            self._save_run_timestamps()

    def _save_run_timestamps(self) -> None:
        """Write the run's frame timings to one parquet named after its end time.

        This strategy keeps no dataset, so unlike the recording strategies there
        is no episode to name the file after and no obvious directory to put it
        in — hence the explicit ``timestamp_log_dir``.
        """
        if self._timestamp_logger is None or not len(self._timestamp_logger):
            return
        name = f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        path = self._timestamp_logger.save_parquet(
            Path(self.config.timestamp_log_dir).expanduser() / name
        )
        logger.info("Saved %d frame timings to %s", len(self._timestamp_logger), path)
        self._timestamp_logger.clear()

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
