"""Load LocoReal / LocoVR pickle trajectories.

Each scene pickle is a list of trajectories.  Each trajectory:
  {time:(T,), goal:(3,), scene_id:str, p1:{part:{pos:(T,3),pose:(T,3)}}, p2:{...}}
Parts: 'head', 'right hand', 'waist'.  pos = (x, y, height) world metres;
pose = euler (roll, pitch, yaw) radians -> heading yaw = pose[:, 2].

We use p1 (the goal-directed walker) and treat p2 as a dynamic distractor we
ignore for the static-layout question (see next-plan.md constraints).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from locovrnf import config as C


LOCOREAL_PICKLES = C.LOCOREAL_DIR / "LocoReal Dataset"
LOCOVR_PICKLES = C.LOCOVR_DIR


def load_scene(scene_id: int | str, pickle_dir: Path = None) -> list:
    pickle_dir = pickle_dir or LOCOREAL_PICKLES
    with open(Path(pickle_dir) / f"{int(scene_id):03d}", "rb") as f:
        return pickle.load(f)


def part_xy(traj: dict, person: str = "p1", part: str = "waist") -> np.ndarray:
    """(T, 2) world floor (x, y) metres for one body part."""
    return np.asarray(traj[person][part]["pos"], dtype=np.float32)[:, :2]


def part_yaw(traj: dict, person: str = "p1", part: str = "waist") -> np.ndarray:
    """(T,) heading yaw (radians) = euler-z of the pose."""
    return np.asarray(traj[person][part]["pose"], dtype=np.float32)[:, 2]


def goal_xy(traj: dict) -> np.ndarray:
    """(2,) goal floor (x, y); handles LocoReal (3,) and LocoVR (1,3)."""
    return np.asarray(traj["goal"], dtype=np.float32).reshape(-1)[:2]


def heading_from_motion(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(T,) heading yaw from velocity (0 = +y, CCW toward +x); robust to quat
    convention differences between LocoReal (euler) and LocoVR (quat)."""
    v = np.zeros_like(xy)
    v[1:] = xy[1:] - xy[:-1]
    v[0] = v[1]
    yaw = np.arctan2(v[:, 0], v[:, 1])          # atan2(dx, dy): 0=+y toward +x
    slow = np.linalg.norm(v, axis=1) < eps
    for i in np.flatnonzero(slow):              # carry heading through pauses
        yaw[i] = yaw[i - 1] if i else yaw[i]
    return yaw.astype(np.float32)


def all_waist_xy(scene_id: int | str, person: str = "p1",
                 part: str = "waist", pickle_dir: Path = None) -> np.ndarray:
    """Stack every frame's (x, y) from every trajectory in a scene -> (N, 2)."""
    data = load_scene(scene_id, pickle_dir)
    return np.concatenate([part_xy(t, person, part) for t in data], axis=0)


def scene_ids_locoreal() -> list[int]:
    return sorted(int(p.name) for p in LOCOREAL_PICKLES.iterdir() if p.name.isdigit())


def scene_ids_locovr() -> list[int]:
    return sorted(int(p.name) for p in LOCOVR_PICKLES.iterdir()
                  if p.is_file() and p.name.isdigit())
