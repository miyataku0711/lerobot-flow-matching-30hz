#!/usr/bin/env python

# Local addition (2026-08-05)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import TIMESTAMP_KEY_PREFIX, TIMESTAMP_OBSERVATION_NAME


class TimestampLogger:
    """
    Timestamp logger for recording data acquisition timing.

    This logger is intentionally independent from the LeRobot dataset format.

    One record corresponds to one recorded frame.

    Example record:

    {
        "frame_index": 123,

        "loop_start_timestamp": 5142.123456,

        "loop_control_end_timestamp": 5142.123456,

        "loop_end_timestamp": 5142.123456,

        "observation_timestamp": 5142.123456,

        "action_timestamp": 5142.132211,

        "front_timestamp": 5142.109876,

        "wrist_timestamp": 5142.118765,
    }
    """

    def __init__(self):
        self.records: list[dict] = []

    def clear(self) -> None:
        """Clear all buffered records."""
        self.records.clear()

    def add(
        self,
        frame_index: int,
        loop_start_timestamp: float,
        loop_control_end_timestamp: float,
        loop_end_timestamp: float,
        observation_timestamp: float,
        action_timestamp: float,
        camera_timestamps: dict[str, float] | None = None,
        extra_fields: dict | None = None,
    ) -> None:
        """
        Add a single frame timestamp entry.

        Parameters
        ----------
        frame_index:
            Dataset frame index inside episode.

        loop_start_timestamp:
            Timestamp recorded immediately after loop_record() starts.

        loop_control_end_timestamp:
            Timestamp recorded immediately after get_observation(), get_action() and send_action() ends.

        loop_end_timestamp:
            Timestamp recorded immediately before loop_record() ends. (This timestamp includes visualization time.)

        observation_timestamp:
            Timestamp recorded immediately before/after
            robot.get_observation().

        action_timestamp:
            Timestamp recorded immediately before/after
            teleop.get_action().

        camera_timestamps:
            Arbitrary camera timestamps.

            Example:
            {
                "front_timestamp": 123.456,
                "wrist_timestamp": 123.457,
            }

        extra_fields:
            Optional additional fields for future use.

            Example:
            {
                "quest_pose_timestamp": ...
            }
        """

        row = {
            "frame_index": frame_index,
            "loop_start_timestamp": loop_start_timestamp,
            "loop_control_end_timestamp": loop_control_end_timestamp,
            "loop_end_timestamp": loop_end_timestamp,
            "observation_timestamp": observation_timestamp,
            "action_timestamp": action_timestamp,
        }

        if camera_timestamps:
            row.update(camera_timestamps)

        if extra_fields:
            row.update(extra_fields)

        self.records.append(row)

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all buffered records to DataFrame."""

        if len(self.records) == 0:
            return pd.DataFrame()

        return pd.DataFrame(self.records)

    def save_parquet(self, filepath: str | Path) -> Path:
        """
        Save records to parquet file.

        Parameters
        ----------
        filepath:
            Target parquet path.

        Returns
        -------
        Path
            Saved path.
        """

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_parquet(filepath, index=False)

        return filepath

    def save_csv(self, filepath: str | Path) -> Path:
        """
        Save records to csv file.

        Useful for quick inspection in Excel.
        """

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

        return filepath

    def get_statistics(self) -> dict:
        """
        Return only basic statistics.

        Heavy analysis should be done later
        using the saved parquet files.
        """

        stats = {
            "num_frames": len(self.records),
        }

        return stats


def split_observation_timestamps(
    observation: dict,
) -> tuple[float | None, dict[str, float]]:
    """
    Pull the ``__timestamp__*`` entries out of a raw robot observation.

    Robots that support timing (see ``so_follower``) attach
    ``__timestamp__observation`` and ``__timestamp__<camera>`` keys to the
    observation dict. This splits them into the observation timestamp and a
    ``{"<camera>_timestamp": value}`` mapping shaped for :meth:`TimestampLogger.add`.

    Must be given the *raw* observation: the processor pipeline drops these keys.

    The producing side names these keys from the same two constants, so the
    convention lives in one place (``lerobot.utils.constants``).

    Returns
    -------
    tuple
        ``(observation_timestamp, camera_timestamps)``. The first element is
        ``None`` when the robot does not report an observation timestamp.
    """

    observation_timestamp: float | None = None
    camera_timestamps: dict[str, float] = {}

    for key, value in observation.items():
        if not key.startswith(TIMESTAMP_KEY_PREFIX):
            continue

        name = key.removeprefix(TIMESTAMP_KEY_PREFIX)

        if name == TIMESTAMP_OBSERVATION_NAME:
            observation_timestamp = value
        else:
            camera_timestamps[f"{name}_timestamp"] = value

    return observation_timestamp, camera_timestamps
