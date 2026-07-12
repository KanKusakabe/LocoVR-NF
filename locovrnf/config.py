"""Shared paths, map spec, and occupancy utilities for LocoVR / LocoReal.

Coordinate convention (from the official loader, data/locoreal/vis_traj.py):
  world floor plane = (x, y) metres; person pos is (T, 3) = (x, y, height).
  map image is size_px x size_px for size_m x size_m metres, centred at origin:
      px_col = (x + size_m/2) * size_px / size_m
      px_row = (y + size_m/2) * size_px / size_m
  so metres in [-5, +5] map to pixels [0, 1024]; 1 px ~ 0.977 cm.
  heading yaw for a body part = pose[:, 2] (euler z, radians) in LocoReal.

Binary map polarity is verified empirically in coordcheck (trajectories must sit
on the walkable value). OCC_IS_DARK records the result: occupied = (img < 0.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOCOREAL_DIR = DATA / "locoreal"
LOCOVR_DIR = DATA / "locovr"
MAPS_DIR = DATA / "maps"
TESTCODE_DIR = DATA / "testcode"
RESULTS = ROOT / "results"
PROC = DATA / "processed"

# --- map geometry ---
SIZE_M = 10.0                     # map covers [-5, +5] m on each floor axis
SIZE_PX = 1024
PX_PER_M = SIZE_PX / SIZE_M       # 102.4 px / m
M_PER_PX = SIZE_M / SIZE_PX       # ~0.977 cm / px
MAP_MIN = -SIZE_M / 2.0           # -5.0 m (world coord at pixel 0)
MAP_MAX = SIZE_M / 2.0            # +5.0 m

# occupancy polarity of the binary maps (occupied cells are the DARK value 0).
# Verified in coordcheck: trajectories sit on the bright (=1) walkable value.
OCC_IS_DARK = True

# --- ego-centric occupancy crop fed to the affordance CNN (matches Layout-NF) ---
CROP_PX = 48                      # 48 x 48 cells
CROP_M = 4.8                      # covering 4.8 m x 4.8 m  (10 cm / cell)

PARTS = ("head", "right hand", "waist")


def m2px(pos_m: np.ndarray) -> np.ndarray:
    """(x, y) or (...,2/3) metres -> pixel (col, row); extra dims pass through."""
    pos_m = np.asarray(pos_m, dtype=np.float32)
    out = pos_m.copy()
    out[..., 0] = (pos_m[..., 0] + SIZE_M / 2) * PX_PER_M
    out[..., 1] = (pos_m[..., 1] + SIZE_M / 2) * PX_PER_M
    return out


def load_occupancy(scene_id: int | str, maps_dir: Path = None) -> np.ndarray:
    """Return [SIZE_PX, SIZE_PX] float32 occupancy, 1 = occupied (furniture/wall).

    Indexed as occ[row, col] with row along +y, col along +x (world floor).
    """
    import matplotlib.image as mpimg
    maps_dir = maps_dir or (LOCOREAL_DIR / "binary_map")
    img = mpimg.imread(str(Path(maps_dir) / f"{int(scene_id):03d}.png"))
    a = img if img.ndim == 2 else img[..., 0]
    occ = (a < 0.5) if OCC_IS_DARK else (a >= 0.5)
    return occ.astype(np.float32)


def occ_to_map_tensor(occ: np.ndarray, device) -> torch.Tensor:
    """Occupancy [row(y), col(x)] -> [x, y] map tensor for ego_crops (x-first)."""
    return torch.tensor(occ.T, device=device)   # transpose so dim0=x(col), dim1=y(row)


def ego_crops(map_xy: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor,
              crop_px: int = CROP_PX, crop_m: float = CROP_M,
              ahead: float = 0.0) -> torch.Tensor:
    """Ego-centric rotated occupancy crops via one batched grid_sample.

    map_xy : [SIZE_PX, SIZE_PX] float, dim0 = x, dim1 = y (see occ_to_map_tensor)
    xy     : [B, 2] world (x, y) metres
    yaw    : [B] heading in the (x, y) plane (0 = +y forward, CCW toward +x)
    returns: [B, crop_px, crop_px] float in {0,1}; row 0 = ahead of person
    """
    B = xy.shape[0]
    dev = xy.device
    half = crop_m / 2.0
    lin = torch.linspace(-half, half, crop_px, device=dev)
    vv, uu = torch.meshgrid(lin + ahead, lin, indexing="ij")   # v=forward, u=lateral
    fwd = torch.stack([torch.sin(yaw), torch.cos(yaw)], -1)     # [B,2] (x,y)
    rgt = torch.stack([torch.cos(yaw), -torch.sin(yaw)], -1)    # [B,2]
    pts = (xy[:, None, None, :]
           + vv[None, :, :, None] * fwd[:, None, None, :]
           + uu[None, :, :, None] * rgt[:, None, None, :])       # [B,P,P,2] (x,y)
    gx = (pts[..., 0] - MAP_MIN) / (MAP_MAX - MAP_MIN) * 2 - 1   # x -> dim0
    gy = (pts[..., 1] - MAP_MIN) / (MAP_MAX - MAP_MIN) * 2 - 1   # y -> dim1
    grid = torch.stack([gy, gx], dim=-1)                         # grid_sample wants (W,H)
    img = map_xy[None, None].expand(B, -1, -1, -1)
    out = torch.nn.functional.grid_sample(
        img, grid, mode="nearest", padding_mode="border", align_corners=True)
    return out[:, 0].flip(1)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print("wrote", path)
