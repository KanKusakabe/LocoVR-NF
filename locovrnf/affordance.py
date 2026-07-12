"""C1 - real-furniture avoidance scoring (the real-data version of Layout-NF A2).

Train the spatial-affordance flow  p(visit offset | local occupancy patch)  on
LocoReal p1 waist positions, using the *real* binary occupancy of each of the 4
physical layouts.  Then test whether the learned density actually treats real
furniture as low-affordance:

  Gate A (model):  logp of a visit centred on a real furniture cell is lower
                   than on a walkable free cell (AUC + per-layout sign test).
  Gate B (data):   cells that are furniture in some layouts but free in others
                   show a large measured visit-density drop when occupied
                   (a natural cross-layout counterfactual, no model needed).

Run:  uv run python -m locovrnf.affordance
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C
from locovrnf import dataio as D
from locovrnf.model import AffordanceFlow

JITTER = 1.2            # patch-centre jitter (m); LocoReal rooms are ~4-5 m
SUB = 3                 # subsample frames (~90 Hz -> 30 Hz)
SEED = 0
SCALE = C.CROP_M / 2.0  # offset units: patch half-width -> [-1,1]


def px2m(col, row):
    return col * C.M_PER_PX + C.MAP_MIN, row * C.M_PER_PX + C.MAP_MIN


def build_samples(rng):
    scenes = D.scene_ids_locoreal()
    pos, scene = [], []
    for sid in scenes:
        data = D.load_scene(sid)
        xy = np.concatenate([D.part_xy(t, "p1", "waist")[::SUB] for t in data], 0)
        pos.append(xy.astype(np.float32))
        scene.append(np.full(len(xy), sid, np.int64))
    pos = np.concatenate(pos); scene = np.concatenate(scene)
    split = (rng.random(len(pos)) < 0.15).astype(np.int8)
    return pos, scene, split, scenes


def ego_offset(pos, centers, angs):
    d = pos - centers
    sy, cy = np.sin(angs), np.cos(angs)
    fwd = d[:, 0] * sy + d[:, 1] * cy
    lat = d[:, 0] * cy - d[:, 1] * sy
    return np.stack([lat, fwd], -1)


def crops_for(map_xy, centers, angs, dev):
    return C.ego_crops(map_xy, torch.tensor(centers, device=dev),
                       torch.tensor(angs, device=dev), ahead=0.0)


def furniture_free_cells(occ, visits_px, rng, k=3000, pad=6):
    """Sample furniture vs free pixel cells inside the walkable bounding box."""
    col, row = visits_px[:, 0].astype(int), visits_px[:, 1].astype(int)
    c0, c1 = col.min() - pad, col.max() + pad
    r0, r1 = row.min() - pad, row.max() + pad
    box = np.zeros_like(occ, bool); box[r0:r1, c0:c1] = True
    furn = np.argwhere(box & (occ > 0.5))      # (row, col) occupied inside room
    free = np.argwhere(box & (occ < 0.5))
    fi = furn[rng.permutation(len(furn))[:k]]
    ff = free[rng.permutation(len(free))[:k]]
    return fi[:, ::-1].astype(np.float32), ff[:, ::-1].astype(np.float32)  # -> (col,row)


def auc_of(pos_scores, neg_scores):
    lab = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    sc = np.r_[pos_scores, neg_scores]
    o = np.argsort(sc); r = np.empty_like(o, float); r[o] = np.arange(len(sc))
    return float((r[lab == 1].sum() - len(pos_scores) * (len(pos_scores) - 1) / 2)
                 / (len(pos_scores) * len(neg_scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    pos, scene, split, scenes = build_samples(rng)
    occs = {sid: C.load_occupancy(sid) for sid in scenes}
    maps = {sid: C.occ_to_map_tensor(occs[sid], dev) for sid in scenes}
    print(f"samples {len(pos)} across {len(scenes)} layouts")

    model = AffordanceFlow().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def run_epoch(sel, train):
        order = rng.permutation(sel); tot = cnt = 0; model.train(train)
        for sid in scenes:
            m = order[scene[order] == sid]
            for i in range(0, len(m), args.batch):
                b = m[i:i + args.batch]
                centers = pos[b] + rng.uniform(-JITTER, JITTER, (len(b), 2)).astype(np.float32)
                angs = rng.uniform(0, 2 * np.pi, len(b)).astype(np.float32)
                crop = crops_for(maps[sid], centers, angs, dev)
                x = torch.tensor(ego_offset(pos[b], centers, angs) / SCALE, device=dev)
                loss = -model.log_prob(crop, x).mean()
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(b); cnt += len(b)
        return tot / max(cnt, 1)

    tr_sel = np.flatnonzero(split == 0); va_sel = np.flatnonzero(split == 1)
    t0 = time.time()
    for ep in range(args.epochs):
        tr = run_epoch(tr_sel, True)
        with torch.no_grad():
            va = run_epoch(va_sel, False)
        print(f"ep{ep} train {tr:.4f} | val {va:.4f}  (uniform=1.386)")
    print(f"trained in {time.time()-t0:.0f}s")

    # ---- Gate A: model logp on real furniture vs free cells ----
    def score_cells(sid, cells_px):
        col, row = cells_px[:, 0], cells_px[:, 1]
        x, y = px2m(col, row)
        centers = np.stack([x, y], -1).astype(np.float32)
        angs = rng.uniform(0, 2 * np.pi, len(centers)).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(centers), 4096):
                crop = crops_for(maps[sid], centers[i:i + 4096], angs[i:i + 4096], dev)
                xz = torch.zeros((len(crop), 2), device=dev)   # offset 0 = at cell
                out.append(model.log_prob(crop, xz).cpu().numpy())
        return np.concatenate(out)

    gateA = {}; lp_f_all = []; lp_o_all = []
    for sid in scenes:
        vpx = C.m2px(pos[scene == sid])
        furn_px, free_px = furniture_free_cells(occs[sid], vpx, rng)
        lp_o = score_cells(sid, furn_px)   # on furniture
        lp_f = score_cells(sid, free_px)   # on free
        lp_o_all.append(lp_o); lp_f_all.append(lp_f)
        gateA[f"{sid:03d}"] = {
            "logp_furniture_mean": round(float(lp_o.mean()), 4),
            "logp_free_mean": round(float(lp_f.mean()), 4),
            "furniture_lower": bool(np.median(lp_o) < np.median(lp_f)),
            "auc_free_over_furniture": round(auc_of(lp_f, lp_o), 4),
        }
        print(f"  layout {sid:03d}: logp furniture {lp_o.mean():+.3f} "
              f"free {lp_f.mean():+.3f} | AUC {gateA[f'{sid:03d}']['auc_free_over_furniture']:.3f}")
    lp_f_all = np.concatenate(lp_f_all); lp_o_all = np.concatenate(lp_o_all)
    sign_pass = all(v["furniture_lower"] for v in gateA.values())
    auc_overall = auc_of(lp_f_all, lp_o_all)

    # ---- Gate B: cross-layout empirical visit-density counterfactual ----
    dens = {}
    for sid in scenes:
        vpx = C.m2px(pos[scene == sid]).astype(int)
        h = np.zeros((C.SIZE_PX, C.SIZE_PX), np.float32)
        np.add.at(h, (np.clip(vpx[:, 1], 0, C.SIZE_PX - 1),
                      np.clip(vpx[:, 0], 0, C.SIZE_PX - 1)), 1.0)
        from scipy.ndimage import gaussian_filter
        dens[sid] = gaussian_filter(h, 8) / max(len(vpx), 1)
    occ_stack = np.stack([occs[s] for s in scenes])          # (4,H,W)
    den_stack = np.stack([dens[s] for s in scenes])
    varies = (occ_stack.max(0) > 0.5) & (occ_stack.min(0) < 0.5)   # cell furniture in some, free in others
    free_mask = varies[None] & (occ_stack < 0.5)
    occ_mask = varies[None] & (occ_stack > 0.5)
    d_free = float(den_stack[free_mask].mean())
    d_occ = float(den_stack[occ_mask].mean())
    ratio = d_free / max(d_occ, 1e-9)
    print(f"  Gate B cross-layout: density free-state {d_free:.2e} vs "
          f"furniture-state {d_occ:.2e}  ->  {ratio:.1f}x drop "
          f"({int(varies.sum())} varying cells)")

    result = {
        "n_samples": int(len(pos)), "n_layouts": len(scenes),
        "val_nll": round(run_epoch(va_sel, False), 4), "uniform_nll": round(float(np.log(4)), 4),
        "gateA_model": {"per_layout": gateA, "sign_test_pass": bool(sign_pass),
                        "auc_free_over_furniture_overall": round(auc_overall, 4)},
        "gateB_data": {"density_free_state": d_free, "density_furniture_state": d_occ,
                       "drop_ratio": round(ratio, 2), "n_varying_cells": int(varies.sum())},
        "gate_pass": bool(sign_pass and auc_overall > 0.65 and ratio > 3.0),
    }
    C.save_json(result, C.RESULTS / "c1_affordance.json")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].hist(lp_f_all, 60, alpha=.6, color="#2563eb", label="free cells", density=True)
    ax[0].hist(lp_o_all, 60, alpha=.6, color="#c2410c", label="furniture cells", density=True)
    ax[0].set_title(f"Gate A · visit logp: furniture vs free\nAUC {auc_overall:.3f} "
                    f"(1=perfect separation)"); ax[0].set_xlabel("model logp at cell")
    ax[0].legend()
    ax[1].bar(["free-state", "furniture-state"], [d_free, d_occ],
              color=["#2563eb", "#c2410c"])
    ax[1].set_title(f"Gate B · same cell, cross-layout\nvisit density {ratio:.1f}x drop "
                    f"when furniture present")
    ax[1].set_ylabel("mean visit density")
    sid0 = scenes[0]
    ax[2].imshow(occs[sid0], cmap="gray_r")
    ax[2].imshow(dens[sid0], cmap="hot", alpha=.6)
    ax[2].set_title(f"layout {sid0:03d}: visit density (hot) vs furniture (dark)")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.tight_layout(); out = C.RESULTS / "c1_affordance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)

    (C.RESULTS / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, C.RESULTS / "ckpt" / "affordance.pt")
    print("C1 gate_pass:", result["gate_pass"])


if __name__ == "__main__":
    main()
