"""Register each LocoVR home's world coordinates to its 10 m occupancy map.

LocoVR maps are per-home 10 m x 10 m crops of HM3D floors, so (unlike LocoReal)
the fixed (x+5)*1024/10 transform does not apply -- each home has its own,
undocumented world origin.  We recover that origin from a physical constraint:
the trajectories must lie on walkable (free) space.  A coarse+fine 2D offset
search maximises the fraction of p1 waist samples that fall on free cells; the
resulting per-home offset (x0, y0) maps world (x, y) -> pixel via
    col = (x - x0) / res ,  row = (y - y0) / res .

This is the LocoVR analogue of C0 (coordinate integrity).  Homes that cannot be
aligned above MIN_QUALITY (e.g. span > 10 m or a corrupt map) are dropped.
Offsets are cached to data/processed/locovr_offsets.json.

Run:  uv run python -m locovrnf.register
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C
from locovrnf import dataio as D

RES = C.M_PER_PX
MIN_QUALITY = 0.90
CACHE = C.PROC / "locovr_offsets.json"


def _onfree(occ, xy, x0, y0):
    col = ((xy[:, 0] - x0) / RES).astype(int)
    row = ((xy[:, 1] - y0) / RES).astype(int)
    ok = (col >= 0) & (col < C.SIZE_PX) & (row >= 0) & (row < C.SIZE_PX)
    if ok.sum() == 0:
        return 0.0
    free = occ[np.clip(row, 0, C.SIZE_PX - 1), np.clip(col, 0, C.SIZE_PX - 1)] < 0.5
    return float((free & ok).mean())          # counts off-map samples as failures


def register_home(hid, occ=None, xy=None):
    if occ is None:
        occ = C.load_occupancy(hid, maps_dir=C.MAPS_DIR / "binary_map")
    if xy is None:
        xy = np.concatenate([D.part_xy(t, "p1", "waist")
                             for t in D.load_scene(hid, D.LOCOVR_PICKLES)], 0)
    sub = xy[:: max(1, len(xy) // 3000)]
    span_m = C.SIZE_M
    xs = np.arange(xy[:, 0].max() - span_m, xy[:, 0].min() + 1e-3, 0.5)
    ys = np.arange(xy[:, 1].max() - span_m, xy[:, 1].min() + 1e-3, 0.5)
    if len(xs) == 0 or len(ys) == 0:          # trajectory spans > 10 m: cannot fit
        return None, 0.0
    best = (-1.0, (xs[0], ys[0]))
    for x0 in xs:
        for y0 in ys:
            s = _onfree(occ, sub, x0, y0)
            if s > best[0]:
                best = (s, (x0, y0))
    bx, by = best[1]                          # fine refine +/- 0.5 m at 0.1 m
    for x0 in np.arange(bx - 0.5, bx + 0.5, 0.1):
        for y0 in np.arange(by - 0.5, by + 0.5, 0.1):
            s = _onfree(occ, sub, x0, y0)
            if s > best[0]:
                best = (s, (x0, y0))
    q = _onfree(occ, xy, *best[1])            # final quality on all samples
    return (float(best[1][0]), float(best[1][1])), q


def main():
    import json
    homes = D.scene_ids_locovr()
    bmaps = set(int(p.stem) for p in (C.MAPS_DIR / "binary_map").glob("*.png"))
    offsets, quals, dropped = {}, [], []
    for hid in homes:
        if hid not in bmaps:
            dropped.append(hid); continue
        off, q = register_home(hid)
        if off is None or q < MIN_QUALITY:
            dropped.append(hid); continue
        offsets[str(hid)] = {"x0": off[0], "y0": off[1], "on_free": round(q, 4)}
        quals.append(q)
    C.PROC.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(offsets, indent=1))
    quals = np.array(quals)
    print(f"registered {len(offsets)}/{len(homes)} homes | dropped {len(dropped)} "
          f"| on-free median {np.median(quals):.3f} mean {quals.mean():.3f} min {quals.min():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(quals, 30, color="#2563eb")
    ax[0].axvline(MIN_QUALITY, color="r", ls="--")
    ax[0].set_title(f"C2 coordinate integrity · LocoVR\n{len(offsets)} homes registered, "
                    f"median on-free {np.median(quals)*100:.1f}%")
    ax[0].set_xlabel("fraction of trajectory on free space")
    # show one aligned home
    hid = max(offsets, key=lambda h: offsets[h]["on_free"])
    occ = C.load_occupancy(hid, maps_dir=C.MAPS_DIR / "binary_map")
    xy = np.concatenate([D.part_xy(t, "p1", "waist")
                        for t in D.load_scene(int(hid), D.LOCOVR_PICKLES)], 0)
    x0, y0 = offsets[hid]["x0"], offsets[hid]["y0"]
    col = (xy[:, 0] - x0) / RES; row = (xy[:, 1] - y0) / RES
    ax[1].imshow(occ, cmap="gray_r"); ax[1].plot(col, row, ".", ms=0.5, color="#c2410c", alpha=.3)
    ax[1].set_title(f"home {hid}: registered trajectories on occupancy")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.tight_layout()
    out = C.RESULTS / "c2_register.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out, "and", CACHE)


if __name__ == "__main__":
    main()
