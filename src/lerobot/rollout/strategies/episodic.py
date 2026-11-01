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

"""Episodic rollout strategy: mirrors the behavior of ``lerobot-record``.

- Policy drives the robot during each recording episode.
- An optional teleoperator can drive the robot during reset phases so the
  operator can bring the environment back to its starting configuration.
  If no teleop is connected the robot stays in its current position.
- Keyboard controls:

      Right arrow  — end the current episode or reset phase early
      Left arrow   — discard the current episode and re-record it
      Escape       — stop the recording session

Dataset naming follows the rollout convention: repo names must start with ``rollout_``.
"""

from __future__ import annotations

import contextlib
import logging
import time

from lerobot.common.control_utils import (
    follower_smooth_move_to,
    teleop_smooth_move_to,
    teleop_supports_feedback,
)
from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import LoopClock
from lerobot.utils.timestamp_logger import TimestampLogger
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_visualization_data

from ..configs import EpisodicStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, safe_push_to_hub, send_next_action

logger = logging.getLogger(__name__)


class EpisodicStrategy(RolloutStrategy):
    """Policy-driven multi-episode recording, mirrors the behavior of ``lerobot-record``.

    Each recording episode runs the policy for maximum ``dataset.episode_time_s``
    seconds, recording every frame.  A reset phase of ``dataset.reset_time_s``
    follows every episode (except the last) so the operator can manually
    reset the environment.  During the reset phase, an optional teleoperator
    drives the robot; if none is present the robot returns to its initial joint positions captured at startup.

    The policy state (hidden state, RTC queue, interpolator) is reset at
    the start of each recording episode.

    Keyboard events:
        right arrow  → end current episode or reset phase early
        left arrow   → discard & re-record current episode
        ESC          → stop the session
    """

    config: EpisodicStrategyConfig

    def __init__(self, config: EpisodicStrategyConfig) -> None:
        super().__init__(config)
        self._listener = None
        self._events: dict | None = None

    def setup(self, ctx: RolloutContext) -> None:
        """Start the inference engine and attach the keyboard listener."""
        self._init_engine(ctx)
        self._listener, self._events = init_keyboard_listener()
        logger.info("Episodic strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Main multi-episode recording loop."""
        cfg = ctx.runtime.cfg
        dataset_cfg = cfg.dataset
        robot = ctx.hardware.robot_wrapper
        teleop = ctx.hardware.teleop
        dataset = ctx.data.dataset
        events = self._events
        features = ctx.data.dataset_features

        fps = cfg.fps
        episode_time_s = dataset_cfg.episode_time_s
        reset_time_s = dataset_cfg.reset_time_s
        num_episodes = dataset_cfg.num_episodes
        single_task = dataset_cfg.single_task or cfg.task
        play_sounds = cfg.play_sounds

        # Frame-level timings for control-period jitter analysis, written per
        # episode to <dataset root>/timestamp_logs/ as in lerobot-record.
        self._timestamp_logger = TimestampLogger()

        display_compressed = (
            True
            if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
            else cfg.display_compressed_images
        )

        # Local change (2026-09-28): run one inference before the first episode and discard it.
        # The first inference of the process is ~0.6 s (CUDA/cuDNN initialisation); inside the timed
        # loop it stalled the first tick of episode 0 and dropped ~20 frames. The action is not sent,
        # and the engine is reset at the start of every episode below.
        if not cfg.use_torch_compile:
            obs_frame = build_dataset_frame(features, self._process_observation_and_notify(
                ctx.processors, robot.get_observation()), prefix=OBS_STR)
            self._engine.get_action(obs_frame)
        # Local change (2026-10-02): with torch.compile, compile and
        # record CUDA graphs here instead of in the first episode.
        self._prewarm_compiled_inference(ctx)

        with VideoEncodingManager(dataset):
            try:
                recorded_episodes = 0
                while recorded_episodes < num_episodes and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    # Reset policy state at episode start (discard leftover hidden state / queue)
                    self._engine.reset()
                    self._interpolator.reset()
                    self._engine.resume()

                    log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                    self._policy_loop(
                        ctx=ctx,
                        robot=robot,
                        events=events,
                        features=features,
                        fps=fps,
                        control_time_s=episode_time_s,
                        dataset=dataset,
                        single_task=single_task,
                    )

                    # Reset phase, skip after the last episode (but run when re-recording)
                    if not events["stop_recording"] and (
                        recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", play_sounds)

                        if teleop:
                            # Smooth handover so the transition to teleop control is jerk-free.
                            # For actuated teleops: drive the leader arm to the follower's current
                            # position so the operator takes over without fighting the arm.
                            # For non-actuated teleops: slide the follower to the teleop's current
                            # pose instead, since the leader cannot be driven.
                            # Disabled entirely with --strategy.smooth_handover=false (useful for
                            # clutch-style teleops that re-reference at the current robot pose on
                            # engage).
                            if self.config.smooth_handover:
                                obs = robot.get_observation()
                                current_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
                                if (
                                    teleop_supports_feedback(teleop)
                                    and self.config.smooth_leader_to_follower_handover
                                ):
                                    logger.info("Smooth handover: moving leader arm to follower position")
                                    teleop_smooth_move_to(teleop, current_pos, duration_s=2)
                                    teleop.disable_torque()
                                else:
                                    logger.info("Smooth handover: sliding follower to teleop position")
                                    teleop_action = teleop.get_action()
                                    processed = ctx.processors.teleop_action_processor((teleop_action, obs))
                                    target = ctx.processors.robot_action_processor((processed, obs))
                                    follower_smooth_move_to(robot, current_pos, target, duration_s=1)

                        elif self.config.reset_to_initial_position:
                            # No teleop: return the robot to its startup position.
                            self._return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        self._reset_loop(
                            ctx=ctx,
                            robot=robot,
                            teleop=teleop,
                            events=events,
                            fps=fps,
                            control_time_s=reset_time_s,
                            display_data=cfg.display_data,
                            display_mode=cfg.display_mode,
                            display_compressed=display_compressed,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        self._timestamp_logger.clear()

                        # returns to its initial joint positions captured at startup
                        if not teleop and self.config.reset_to_initial_position:
                            self._return_to_initial_position(hw=ctx.hardware, duration_s=1)

                        continue

                    # Read the index before saving: save_episode() advances it.
                    # Taken from the dataset rather than the local counter so the
                    # filename still lines up with the episode when resuming.
                    episode_index = dataset.num_episodes
                    dataset.save_episode()
                    self._save_episode_timestamps(dataset, episode_index)
                    recorded_episodes += 1
            finally:
                # Save any frames buffered in the current episode so an unexpected
                # exception or KeyboardInterrupt does not silently drop recorded data.
                # suppress: save_episode raises if the buffer is empty (nothing to lose).
                logger.info("Episodic control loop ended — saving any in-progress episode")
                with contextlib.suppress(Exception):
                    episode_index = dataset.num_episodes
                    dataset.save_episode()
                    # Only reached when the episode was actually saved, so an
                    # interrupted session cannot leave an episode without its timings.
                    self._save_episode_timestamps(dataset, episode_index)

    def _policy_loop(
        self,
        ctx: RolloutContext,
        robot,
        events: dict,
        features: dict,
        fps: float,
        control_time_s: float,
        dataset,
        single_task: str,
    ) -> None:
        """Policy-driven recording loop for a single episode.

        Every tick past warmup is timed so control-period jitter can be measured
        afterwards.  Unlike ``lerobot-record``, the reset phase
        is not logged, so a saved episode's timings cover the policy loop only.
        """
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(fps)
        # Local change (2026-09-28): read (and discard) one observation before the timed loop,
        # as in lerobot-record. Right after save_episode() the camera read threads may have been paused
        # (the next episode's encoders were being opened, which holds the GIL) and the cameras then wait
        # up to 0.2 s for a fresh frame. Paying that here keeps the first tick, which always computes a
        # new action chunk after the policy reset, on time.
        robot.get_observation()
        loop_clock = LoopClock(control_interval)

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            target_start = loop_clock.tick()
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(ctx.runtime.cfg.use_torch_compile, loop_start, control_interval):
                continue

            # Spans policy inference, interpolation and send_action — the rollout
            # counterpart of teleop.get_action() in lerobot-record.
            action_timestamp = time.perf_counter()
            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            loop_control_end_timestamp = time.perf_counter()

            if action_dict is not None:
                obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                dataset.add_frame({**obs_frame, **action_frame, "task": single_task})
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            self._record_tick(obs, loop_start, action_timestamp, loop_control_end_timestamp,
                              target_start=target_start, action_sent=action_dict is not None)

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t < 0:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({fps} Hz). "
                    "Dataset frames might be dropped and robot control might be unstable. "
                    "Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long "
                    "3) CPU starvation"
                )
            loop_clock.sleep()
            timestamp = time.perf_counter() - start_t

    def _reset_loop(
        self,
        ctx: RolloutContext,
        robot,
        teleop,
        events: dict,
        fps: float,
        control_time_s: float,
        display_data: bool,
        display_mode: str,
        display_compressed: bool,
    ) -> None:
        """Reset-phase loop: teleop drives the robot if available, no recording."""
        processors = ctx.processors
        control_interval = 1.0 / fps
        loop_clock = LoopClock(control_interval)

        timestamp = 0.0
        start_t = time.perf_counter()

        while timestamp < control_time_s:
            loop_clock.tick()
            loop_start = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            if ctx.runtime.shutdown_event.is_set():
                break

            obs = robot.get_observation()

            if teleop is not None:
                act = teleop.get_action()
                act_teleop = processors.teleop_action_processor((act, obs))
                robot_action = processors.robot_action_processor((act_teleop, obs))
                robot.send_action(robot_action)

                if display_data:
                    obs_processed = processors.robot_observation_processor(obs)
                    log_visualization_data(
                        display_mode,
                        observation=obs_processed,
                        action=act_teleop,
                        compress_images=display_compressed,
                    )

            loop_clock.sleep()
            timestamp = time.perf_counter() - start_t

    def teardown(self, ctx: RolloutContext) -> None:
        """Finalise dataset, stop listener, push to hub, and disconnect hardware."""
        cfg = ctx.runtime.cfg
        play_sounds = cfg.play_sounds

        log_say("Stop recording", play_sounds, blocking=True)

        if self._listener is not None:
            self._listener.stop()

        if ctx.data.dataset is not None:
            logger.info("Finalizing dataset...")
            ctx.data.dataset.finalize()

        if (
            cfg.dataset is not None
            and cfg.dataset.push_to_hub
            and ctx.data.dataset is not None
            and safe_push_to_hub(
                ctx.data.dataset,
                tags=cfg.dataset.tags,
                private=cfg.dataset.private,
            )
        ):
            logger.info("Dataset uploaded to hub")
            log_say("Dataset uploaded to hub", play_sounds)

        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=cfg.return_to_initial_position,
        )
        log_say("Exiting", play_sounds)
        logger.info("Episodic strategy teardown complete")
