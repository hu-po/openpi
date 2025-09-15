"""
Convert the Tatbot LeRobot dataset to an ALOHA-compatible LeRobot dataset.

Why: Training configs in this repo expect ALOHA-style camera names and shapes.
This script remaps Tatbot cameras (e.g., realsense1/2) to ALOHA keys (cam_high/cam_low),
preserves state/action, and writes a fresh LeRobot dataset (images mode) that can be
consumed by LeRobotAlohaDataConfig. Optionally pushes the result to the Hugging Face Hub.

Usage:
  uv run python scripts/convert_tatbot_to_lerobot_aloha.py \
    --src-repo-id tatbot/wow-2025y-09m-10d-15h-34m-08s \
    --dst-repo-id <your-hf-username>/tatbot_wow_pi05_aloha \
    --push-to-hub

Notes:
  - Writes images (not videos) to simplify conversion and avoid re-encoding.
  - Keeps per-episode task strings; training can set prompt_from_task=True to inject them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import tyro

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


@dataclasses.dataclass
class Args:
    # Source HF dataset (LeRobot v2.x format)
    src_repo_id: str = "tatbot/wow-2025y-09m-10d-15h-34m-08s"
    # Destination HF model repo to create/push (e.g., your-username/name)
    dst_repo_id: str = "tatbot/wow_pi05_aloha"

    # Camera remapping: source keys -> ALOHA keys subset
    # Realsense1 -> cam_high, Realsense2 -> cam_low by default
    cam_map: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {"realsense1": "cam_high", "realsense2": "cam_low"}
    )

    # Force robot_type string in destination (purely informational)
    dst_robot_type: str = "aloha"

    # Push to Hugging Face Hub when done
    push_to_hub: bool = False

    # Subset episodes (e.g., "0:10" or "0,2,5") for quick tests; empty means all
    episodes: str | None = None


def _parse_episodes_arg(episodes: str | None, total: int) -> list[int]:
    if not episodes:
        return list(range(total))
    if ":" in episodes:
        start, end = episodes.split(":", 1)
        s = int(start) if start else 0
        e = int(end) if end else total
        return list(range(s, min(e, total)))
    return [int(x) for x in episodes.split(",") if x.strip()]


def _load_episode_meta(repo_root: Path) -> list[dict]:
    meta_path = repo_root / "meta" / "episodes.jsonl"
    episodes_meta: list[dict] = []
    with meta_path.open("r") as f:
        for line in f:
            episodes_meta.append(json.loads(line))
    return episodes_meta


def _create_empty_dst_dataset(
    dst_repo_id: str,
    fps: int,
    robot_type: str,
) -> LeRobotDataset:
    # Two cameras subset: cam_high, cam_low. Images mode keeps HWC uint8 arrays.
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,)},
        "action": {"dtype": "float32", "shape": (14,)},
        "observation.images.cam_high": {"dtype": "image", "shape": (3, 480, 640)},
        "observation.images.cam_low": {"dtype": "image", "shape": (3, 480, 640)},
    }
    # Create fresh dataset home (overwrites existing local copy if any)
    safe_repo_id = dst_repo_id.lstrip("/")
    dst_root = Path(HF_LEROBOT_HOME) / safe_repo_id
    if dst_root.exists():
        # Avoid accidental merge; callers control destination name
        import shutil

        shutil.rmtree(dst_root)

    return LeRobotDataset.create(
        repo_id=dst_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,  # images mode
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
        video_backend=None,
    )


def convert(args: Args) -> None:
    # Download src dataset metadata; loads or fetches into HF_LEROBOT_HOME
    src_meta = LeRobotDatasetMetadata(args.src_repo_id)
    total_eps = src_meta.total_episodes
    fps = src_meta.fps

    # Read per-episode tasks to preserve prompts
    src_root = Path(HF_LEROBOT_HOME) / args.src_repo_id
    episodes_meta = _load_episode_meta(src_root)
    assert len(episodes_meta) == total_eps, "episodes.jsonl length mismatch"

    # Make destination dataset
    dst_ds = _create_empty_dst_dataset(args.dst_repo_id, fps=fps, robot_type=args.dst_robot_type)

    # Choose which episodes to convert
    ep_indices = _parse_episodes_arg(args.episodes, total_eps)

    # Iterate per-episode to preserve episode boundaries and tasks
    for ep_idx in ep_indices:
        # Load a single-episode view to get frames in order
        src_ds = LeRobotDataset(args.src_repo_id, episodes=[ep_idx])
        n = len(src_ds)
        # Compose episode task string if provided; fallback to generic label
        ep_task = None
        if ep_idx < len(episodes_meta):
            tasks = episodes_meta[ep_idx].get("tasks")
            if isinstance(tasks, list) and tasks:
                ep_task = "; ".join(tasks)
        if ep_task is None:
            ep_task = "tatbot"

        for i in range(n):
            sample = src_ds[i]
            # Map cameras
            images_in = {}
            for src_key, dst_key in args.cam_map.items():
                src_field = f"observation.images.{src_key}"
                if src_field in sample:
                    img = sample[src_field]
                    # Expect CHW or HWC; convert to HWC uint8 for robustness
                    arr = np.asarray(img)
                    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
                        # CHW -> HWC
                        arr = np.moveaxis(arr, 0, -1)
                    if arr.dtype != np.uint8:
                        # assume [0,1] floats or 0..255 ints
                        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)
                    images_in[f"observation.images.{dst_key}"] = arr

            frame = {
                "observation.state": sample["observation.state"],
                "action": sample["action"],
                "task": ep_task,
                **images_in,
            }
            dst_ds.add_frame(frame)

        # Close out episode (per-frame 'task' is already stored in frames)
        dst_ds.save_episode()

    # Finalize image writes before optional push
    try:
        dst_ds._wait_image_writer()
    except Exception:
        pass
    if args.push_to_hub:
        dst_ds.push_to_hub()


if __name__ == "__main__":
    # Expose top-level flags (e.g., --src-repo-id) and pass dataclass to converter.
    convert(tyro.cli(Args))
