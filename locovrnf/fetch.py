"""Fetch LocoVR / LocoReal (ICLR 2025, MIT) from Google Drive (idempotent).

Everything lands under data/ inside this project. Re-running skips folders that
already contain files. Uses `uvx gdown --folder <id>` (old gdown has no
--remaining-ok, so we do NOT pass it).

Folders (Drive IDs from the dataset release):
  LocoReal (physical, 4 layouts, ~62 MB)   1C7VANAopABgg_NgfvWAryb5NmcBgawbL
  LocoVR body (131 homes)                  1gE9P3MSJ6dbgpAt4YbEjZn-8cr4jtdVY
  Maps (binary/height/semantic/texture)    1bUT8aHKJmPwvhUFINHDCNmgfyR1vT33G
  TestCode (global path-pred, pretrained)  10ILf7YTiznbzh5pc8CiHkP3Cvlz5Kt_0

Usage:
  uv run python -m locovrnf.fetch locoreal   # C0/C1 minimal slice
  uv run python -m locovrnf.fetch maps        # occupancy maps (for LocoVR)
  uv run python -m locovrnf.fetch locovr      # full 131-home bodies (C2, large)
  uv run python -m locovrnf.fetch testcode     # baseline path predictor (C3)
"""
from __future__ import annotations

import subprocess
import sys

from locovrnf import config as C

FOLDERS = {
    "locoreal": ("1C7VANAopABgg_NgfvWAryb5NmcBgawbL", C.LOCOREAL_DIR),
    "locovr": ("1gE9P3MSJ6dbgpAt4YbEjZn-8cr4jtdVY", C.LOCOVR_DIR),
    "maps": ("1bUT8aHKJmPwvhUFINHDCNmgfyR1vT33G", C.MAPS_DIR),
    "testcode": ("10ILf7YTiznbzh5pc8CiHkP3Cvlz5Kt_0", C.TESTCODE_DIR),
}


def fetch(key: str) -> None:
    folder_id, dest = FOLDERS[key]
    if dest.exists() and any(dest.rglob("*")):
        n = sum(1 for _ in dest.rglob("*") if _.is_file())
        print(f"[fetch] {key}: {dest} already has {n} files — skip")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {key}: gdown folder {folder_id} -> {dest}")
    subprocess.run(
        ["uvx", "gdown", "--folder", folder_id, "-O", str(dest)],
        check=True,
    )
    print(f"[fetch] {key}: done")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["locoreal"]
    for k in keys:
        if k not in FOLDERS:
            raise SystemExit(f"unknown key {k!r}; choose from {list(FOLDERS)}")
        fetch(k)
