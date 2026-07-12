"""C0 gate: do LocoReal trajectories sit on the walkable map and avoid furniture?

For each of the 4 layouts, overlay every p1 waist trajectory on the binary
occupancy map and measure the fraction of trajectory samples that land on
free (non-occupied) cells.  A clean dataset should have the vast majority of
body positions on free space, with furniture/wall cells avoided.

Run:  uv run python -m locovrnf.coordcheck
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C
from locovrnf import dataio as D


def main():
    scenes = D.scene_ids_locoreal()
    fig, axes = plt.subplots(1, len(scenes), figsize=(4 * len(scenes), 4.2))
    if len(scenes) == 1:
        axes = [axes]
    summary = {}

    for ax, sid in zip(axes, scenes):
        occ = C.load_occupancy(sid)                 # [row(y), col(x)], 1=occupied
        data = D.load_scene(sid)
        xy = np.concatenate([D.part_xy(t, "p1", "waist") for t in data], axis=0)
        px = C.m2px(xy)                             # (N,2) = (col, row)
        col = np.clip(px[:, 0].astype(int), 0, C.SIZE_PX - 1)
        row = np.clip(px[:, 1].astype(int), 0, C.SIZE_PX - 1)
        on_occ = occ[row, col]                      # 1 if the person is on an occupied cell
        free_frac = float((on_occ < 0.5).mean())

        ax.imshow(occ, cmap="gray_r", origin="upper", extent=[0, C.SIZE_PX, C.SIZE_PX, 0])
        ax.plot(col, row, ".", ms=0.4, color="#c2410c", alpha=0.25)
        ax.set_title(f"layout {sid:03d}\n{len(data)} traj · "
                     f"{free_frac*100:.1f}% on free space", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        summary[f"{sid:03d}"] = {
            "n_traj": len(data), "n_samples": int(len(xy)),
            "frac_on_free": round(free_frac, 4),
            "occ_dark_frac": round(float(occ.mean()), 4),
        }
        print(f"layout {sid:03d}: {len(data)} traj, {len(xy)} samples, "
              f"{free_frac*100:.1f}% on free space")

    fig.suptitle("C0 · LocoReal p1 waist trajectories on binary occupancy "
                 "(orange = body positions)", fontsize=11)
    fig.tight_layout()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS / "c0_coordcheck.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)

    mean_free = float(np.mean([v["frac_on_free"] for v in summary.values()]))
    result = {"layouts": summary, "mean_frac_on_free": round(mean_free, 4),
              "gate_pass": bool(mean_free > 0.85)}
    C.save_json(result, C.RESULTS / "c0_coordcheck.json")
    print("C0 gate_pass:", result["gate_pass"], "| mean free-space frac:", round(mean_free, 3))


if __name__ == "__main__":
    main()
