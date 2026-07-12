"""C2 - cross-space generalization across 131 real LocoVR homes.

Train the spatial-affordance flow p(visit offset | occupancy patch) on many
real homes and test it on entirely held-out *homes* (not just held-out
trajectories).  This is the real-data, 131-home version of Layout-NF's A4/A7/A8
generalization, and answers the question TROR-MAGNI (1 room) could not:
does the walkable-space density learned from real homes transfer to unseen real
homes?

Two readouts, mirroring A7:
  - held-out-home NLL vs in-distribution NLL (density quality)
  - furniture-vs-free AUC on held-out homes (does furniture still read as
    low-affordance in a home never seen in training)
plus a scale curve: sweep the number of training homes and watch held-out NLL
keep improving while detection AUC saturates early.

Run:  uv run python -m locovrnf.crossspace
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C
from locovrnf import dataio as D
from locovrnf.model import AffordanceFlow

JITTER = 1.2
SUB = 4
SEED = 0
SCALE = C.CROP_M / 2.0
CACHE = C.PROC / "locovr_offsets.json"


def load_offsets():
    return {int(k): (v["x0"], v["y0"]) for k, v in json.loads(CACHE.read_text()).items()}


def build_samples(homes, offsets, max_total=320_000, rng=None):
    """Canonical (map-frame) waist positions per home. canonical = x - x0 - 5,
    so config.ego_crops (MAP_MIN=-5) samples the right patch."""
    pos, home = [], []
    for hid in homes:
        x0, y0 = offsets[hid]
        xy = np.concatenate([D.part_xy(t, "p1", "waist")[::SUB]
                             for t in D.load_scene(hid, D.LOCOVR_PICKLES)], 0)
        xy = xy - np.array([x0, y0], np.float32) - 5.0
        pos.append(xy.astype(np.float32)); home.append(np.full(len(xy), hid))
    pos = np.concatenate(pos); home = np.concatenate(home)
    if len(pos) > max_total:
        idx = (rng or np.random.default_rng(0)).permutation(len(pos))[:max_total]
        pos, home = pos[idx], home[idx]
    return pos, home


def occ_np_of(hid):
    return C.load_occupancy(hid, maps_dir=C.MAPS_DIR / "binary_map")


def ego_offset(pos, centers, angs):
    d = pos - centers
    sy, cy = np.sin(angs), np.cos(angs)
    return np.stack([d[:, 0] * cy - d[:, 1] * sy, d[:, 0] * sy + d[:, 1] * cy], -1)


def auc_of(p, n):
    lab = np.r_[np.ones(len(p)), np.zeros(len(n))]; sc = np.r_[p, n]
    o = np.argsort(sc); r = np.empty_like(o, float); r[o] = np.arange(len(sc))
    return float((r[lab == 1].sum() - len(p) * (len(p) - 1) / 2) / (len(p) * len(n)))


class Trainer:
    def __init__(self, dev, rng, batch=2048):
        self.dev, self.rng, self.batch = dev, rng, batch

    def epoch(self, model, pos, home, homes, occ_cache, opt=None):
        order = self.rng.permutation(len(pos)); tot = cnt = 0
        model.train(opt is not None)
        for hid in homes:
            sel = order[home[order] == hid]
            if len(sel) == 0:
                continue
            mp = C.occ_to_map_tensor(occ_cache[hid], self.dev)
            for i in range(0, len(sel), self.batch):
                b = sel[i:i + self.batch]
                centers = pos[b] + self.rng.uniform(-JITTER, JITTER, (len(b), 2)).astype(np.float32)
                angs = self.rng.uniform(0, 2 * np.pi, len(b)).astype(np.float32)
                crop = C.ego_crops(mp, torch.tensor(centers, device=self.dev),
                                   torch.tensor(angs, device=self.dev), ahead=0.0)
                x = torch.tensor(ego_offset(pos[b], centers, angs) / SCALE, device=self.dev)
                loss = -model.log_prob(crop, x).mean()
                if opt is not None:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(b); cnt += len(b)
        return tot / max(cnt, 1)


def furniture_auc(model, homes, offsets, occ_cache, dev, rng, per_home=400):
    lp_free, lp_furn = [], []
    for hid in homes:
        occ = occ_cache[hid]; x0, y0 = offsets[hid]
        xy = np.concatenate([D.part_xy(t, "p1", "waist")
                            for t in D.load_scene(hid, D.LOCOVR_PICKLES)], 0)
        col = ((xy[:, 0] - x0) / C.M_PER_PX).astype(int)
        row = ((xy[:, 1] - y0) / C.M_PER_PX).astype(int)
        c0, c1 = max(col.min() - 6, 0), min(col.max() + 6, C.SIZE_PX)
        r0, r1 = max(row.min() - 6, 0), min(row.max() + 6, C.SIZE_PX)
        box = np.zeros_like(occ, bool); box[r0:r1, c0:c1] = True
        furn = np.argwhere(box & (occ > 0.5)); free = np.argwhere(box & (occ < 0.5))
        if len(furn) < 20 or len(free) < 20:
            continue
        mp = C.occ_to_map_tensor(occ, dev)
        for cells, out in ((furn, lp_furn), (free, lp_free)):
            sel = cells[rng.permutation(len(cells))[:per_home]]
            # cells are (row,col) px -> canonical world
            cx = sel[:, 1] * C.M_PER_PX + C.MAP_MIN
            cy = sel[:, 0] * C.M_PER_PX + C.MAP_MIN
            centers = np.stack([cx, cy], -1).astype(np.float32)
            angs = rng.uniform(0, 2 * np.pi, len(centers)).astype(np.float32)
            with torch.no_grad():
                crop = C.ego_crops(mp, torch.tensor(centers, device=dev),
                                   torch.tensor(angs, device=dev), ahead=0.0)
                z = torch.zeros((len(crop), 2), device=dev)
                out.append(model.log_prob(crop, z).cpu().numpy())
    lp_free = np.concatenate(lp_free); lp_furn = np.concatenate(lp_furn)
    return auc_of(lp_free, lp_furn), float(lp_furn.mean()), float(lp_free.mean())


def train_model(pos, home, train_homes, occ_cache, dev, rng, epochs):
    model = AffordanceFlow().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = Trainer(dev, rng)
    mask = np.isin(home, train_homes)
    p, h = pos[mask], home[mask]
    for ep in range(epochs):
        tr.epoch(model, p, h, train_homes, occ_cache, opt)
    return model, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=31)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    offsets = load_offsets()
    homes = np.array(sorted(offsets))
    homes = rng.permutation(homes)
    held = homes[:args.holdout]; pool = homes[args.holdout:]
    print(f"{len(homes)} homes | {len(pool)} train-pool | {len(held)} held-out")

    pos, home = build_samples(homes, offsets, rng=rng)
    occ_cache = {hid: occ_np_of(hid) for hid in homes}
    print(f"samples {len(pos)}")

    # ---- scale curve: sweep #train homes, fixed held-out ----
    grid = [n for n in [10, 25, 50, 100] if n <= len(pool)]
    if len(pool) not in grid:
        grid.append(len(pool))
    curve = []
    t0 = time.time()
    for n in grid:
        th = pool[:n]
        model, tr = train_model(pos, home, th, occ_cache, dev, rng, args.epochs)
        with torch.no_grad():
            nll_in = tr.epoch(model, pos[np.isin(home, th)], home[np.isin(home, th)],
                              th, occ_cache)
            nll_out = tr.epoch(model, pos[np.isin(home, held)], home[np.isin(home, held)],
                               held, occ_cache)
        auc, lpf, lpr = furniture_auc(model, held, offsets, occ_cache, dev, rng)
        curve.append({"n_train_homes": int(n), "nll_in_dist": round(nll_in, 4),
                      "nll_held_out": round(nll_out, 4), "furniture_auc_held_out": round(auc, 4)})
        print(f"  n={n:3d} | in-dist NLL {nll_in:.3f} | held-out NLL {nll_out:.3f} "
              f"| furniture AUC {auc:.3f}")
    print(f"scale sweep in {time.time()-t0:.0f}s")

    final = curve[-1]
    res = {
        "n_homes": len(homes), "n_held_out": len(held), "epochs": args.epochs,
        "uniform_nll": round(float(np.log(4)), 4),
        "scale_curve": curve,
        "held_out_nll": final["nll_held_out"], "in_dist_nll": final["nll_in_dist"],
        "furniture_auc_held_out": final["furniture_auc_held_out"],
        "nll_gap_ratio": round(final["nll_held_out"] / final["nll_in_dist"], 3)
                          if final["nll_in_dist"] not in (0,) else None,
        "gate_pass": bool(final["furniture_auc_held_out"] > 0.6
                          and final["nll_held_out"] < 0.5),   # far below uniform 1.386
    }
    C.save_json(res, C.RESULTS / "c2_crossspace.json")

    # ---- figure ----
    ns = [c["n_train_homes"] for c in curve]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(ns, [c["nll_held_out"] for c in curve], "o-", color="#c2410c", label="held-out homes")
    ax[0].plot(ns, [c["nll_in_dist"] for c in curve], "o--", color="#2563eb", label="in-dist")
    ax[0].axhline(np.log(4), color="gray", ls=":", label="uniform")
    ax[0].set_xlabel("# training homes"); ax[0].set_ylabel("NLL (lower=better)")
    ax[0].set_title("C2 · density NLL keeps improving with home diversity"); ax[0].legend()
    ax[1].plot(ns, [c["furniture_auc_held_out"] for c in curve], "o-", color="#16a34a")
    ax[1].axhline(0.5, color="gray", ls=":", label="chance")
    ax[1].set_ylim(0.45, 1.0); ax[1].set_xlabel("# training homes")
    ax[1].set_ylabel("furniture-vs-free AUC (held-out homes)")
    ax[1].set_title("C2 · furniture detection on unseen homes (rises with diversity)"); ax[1].legend()
    fig.tight_layout(); out = C.RESULTS / "c2_scale.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
    print("C2 gate_pass:", res["gate_pass"])


if __name__ == "__main__":
    main()
