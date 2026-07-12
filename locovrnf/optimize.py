"""C5 - furniture-layout optimization: place N pieces to preserve circulation.

Uses the C1 spatial-affordance flow  p(visit | local occupancy)  as the design
objective, exactly as Layout-NF's A3 disruption map: the empty-room walkable
density is a *foot-traffic field* T(x) = exp log p(visit at x | empty-room
patch).  A candidate layout's disruption is the natural traffic its furniture
sits on,

    D(L) = sum over cells covered by furniture of  T(cell) ,

and we search placements (cx, cy, theta per piece) that MINIMISE D — i.e. put
the furniture where people naturally don't go (edges, low-traffic corners) so
the main circulation is least disrupted.  This generalises A3 from one piece to
joint multi-piece optimisation, driven by CEM.  Using the fixed empty-room field
(not re-scoring each layout) keeps the objective non-gameable and fast.

Constraints keep it from the degenerate "stack in a corner" optimum: pieces stay
inside the room and non-overlapping (penalised).

Run:  uv run python -m locovrnf.optimize
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C
from locovrnf.model import AffordanceFlow

SEED = 0
RW, RH = 4.4, 3.2            # room size (m), centred at origin
PIECES = [("sofa", 1.7, 0.75), ("table", 1.15, 0.75), ("cabinet", 1.0, 0.45), ("chair", 0.55, 0.55)]
QSTEP = 0.10                 # traffic-field / coverage grid spacing (m)
CKPT = C.RESULTS / "ckpt" / "affordance.pt"


def base_room():
    col = np.arange(C.SIZE_PX) * C.M_PER_PX + C.MAP_MIN
    x = col[None, :].repeat(C.SIZE_PX, 0); y = col[:, None].repeat(C.SIZE_PX, 1)
    return (~((np.abs(x) < RW / 2) & (np.abs(y) < RH / 2))).astype(np.float32)


def query_points():
    xs = np.arange(-RW / 2 + 0.15, RW / 2 - 0.15, QSTEP)
    ys = np.arange(-RH / 2 + 0.15, RH / 2 - 0.15, QSTEP)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)


def traffic_field(model, dev, base, qpts):
    """T(q) = exp log p(visit at q | empty-room patch), normalised to sum 1."""
    mp = C.occ_to_map_tensor(base, dev); angs = np.zeros(len(qpts), np.float32); out = []
    with torch.no_grad():
        for i in range(0, len(qpts), 4096):
            c = C.ego_crops(mp, torch.tensor(qpts[i:i + 4096], device=dev),
                            torch.tensor(angs[i:i + 4096], device=dev), ahead=0.0)
            z = torch.zeros((len(c), 2), device=dev)
            out.append(model.log_prob(c, z).cpu().numpy())
    logp = np.concatenate(out)
    T = np.exp(logp - logp.max())
    return T / T.sum()


def coverage_count(params, qpts):
    """(Q,) number of furniture pieces covering each query point."""
    cnt = np.zeros(len(qpts), np.int32)
    for i, (_, w, h) in enumerate(PIECES):
        cx, cy, ang = params[i]
        dx = qpts[:, 0] - cx; dy = qpts[:, 1] - cy
        ca, sa = np.cos(ang), np.sin(ang)
        u = dx * ca + dy * sa; v = -dx * sa + dy * ca
        cnt += ((np.abs(u) < w / 2) & (np.abs(v) < h / 2)).astype(np.int32)
    return cnt


def disruption(params, T, qpts, lam=4.0):
    cnt = coverage_count(params, qpts)
    covered = cnt >= 1
    overlap_frac = float(T[cnt >= 2].sum())          # traffic double-covered = pieces overlap
    return float(T[covered].sum()) + lam * overlap_frac


def in_room_clip(params):
    p = params.copy()
    for i, (_, w, h) in enumerate(PIECES):
        r = max(w, h) / 2
        p[i, 0] = np.clip(p[i, 0], -RW / 2 + r, RW / 2 - r)
        p[i, 1] = np.clip(p[i, 1], -RH / 2 + r, RH / 2 - r)
    return p


def cem(T, qpts, rng, iters=30, pop=80, elite=14):
    N = len(PIECES)
    mean = np.zeros((N, 3), np.float32)
    mean[:, 0] = np.linspace(-RW / 2 + 0.9, RW / 2 - 0.9, N)
    std = np.tile(np.array([RW / 3, RH / 3, np.pi / 2], np.float32), (N, 1))
    best = (1e9, None)
    for it in range(iters):
        cands = np.stack([in_room_clip(mean + std * rng.standard_normal((N, 3)).astype(np.float32))
                          for _ in range(pop)])
        scores = np.array([disruption(c, T, qpts) for c in cands])
        idx = np.argsort(scores)                     # ascending: lower disruption is better
        el = cands[idx[:elite]]
        mean = el.mean(0); std = el.std(0) + np.array([0.03, 0.03, 0.05], np.float32)
        if scores[idx[0]] < best[0]:
            best = (float(scores[idx[0]]), cands[idx[0]].copy())
        if it % 5 == 0 or it == iters - 1:
            print(f"  cem it{it:02d} best disruption {best[0]:.4f}")
    return best


def draw(ax, base, params, qpts, T, title):
    from locovrnf.optimize import PIECES as PP
    occ = base.copy()
    for i, (_, w, h) in enumerate(PP):
        cx, cy, ang = params[i]
        # draw furniture rectangle
        ca, sa = np.cos(ang), np.sin(ang)
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        R = np.array([[ca, -sa], [sa, ca]]); world = corners @ R.T + [cx, cy]
        px = C.m2px(world)
        ax.add_patch(plt.Polygon(px, closed=True, color="#3f3a34", zorder=3))
    pq = C.m2px(qpts)
    ax.scatter(pq[:, 0], pq[:, 1], c=T, cmap="hot", s=14, vmin=0, vmax=T.max(), zorder=1)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    m = C.m2px(np.array([[-RW / 2 - .3, -RH / 2 - .3], [RW / 2 + .3, RH / 2 + .3]]))
    ax.set_xlim(m[0, 0], m[1, 0]); ax.set_ylim(m[1, 1], m[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    model = AffordanceFlow().to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev)["state_dict"]); model.eval()

    base = base_room(); q = query_points()
    T = traffic_field(model, dev, base, q)
    print(f"traffic field over {len(q)} cells, {len(PIECES)} pieces")

    rand = in_room_clip(np.stack([[rng.uniform(-RW / 2, RW / 2), rng.uniform(-RH / 2, RH / 2),
                                   rng.uniform(0, np.pi)] for _ in PIECES]).astype(np.float32))
    d_rand = disruption(rand, T, q)
    bestD, bestP = cem(T, q, rng, iters=args.iters)
    print(f"disruption (traffic blocked): naive {d_rand*100:.2f}%  ->  optimized {bestD*100:.2f}%  "
          f"({bestD/d_rand*100:.0f}% of naive)")

    res = {"pieces": [p[0] for p in PIECES], "room_m": [RW, RH], "n_cells": len(q),
           "disruption_naive_pct": round(d_rand * 100, 3), "disruption_optimized_pct": round(bestD * 100, 3),
           "reduction_pct": round((1 - bestD / d_rand) * 100, 1),
           "placement": [{"piece": PIECES[i][0], "cx": round(float(bestP[i, 0]), 3),
                          "cy": round(float(bestP[i, 1]), 3),
                          "deg": round(float(np.degrees(bestP[i, 2]) % 180), 1)} for i in range(len(PIECES))]}
    C.save_json(res, C.RESULTS / "c5_optimize.json")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    draw(ax[0], base, np.array([[9, 9, 0]] * len(PIECES), np.float32), q, T,
         "empty room · foot-traffic field T(x)\n(hot = where people walk)")
    draw(ax[1], base, rand, q, T, f"naive placement\nblocks {d_rand*100:.1f}% of traffic")
    draw(ax[2], base, bestP, q, T,
         f"CEM-optimized (affordance objective)\nblocks {bestD*100:.1f}% "
         f"({(1-bestD/d_rand)*100:.0f}% less than naive)")
    fig.suptitle("C5 · furniture-layout optimization: place N pieces to disrupt the least "
                 "foot-traffic", fontsize=12)
    out = C.RESULTS / "c5_optimize.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
