"""C3 - goal-conditioned trajectory generation = re-design simulation.

Formulation change from the static density p(position | occupancy) to a
goal-conditioned autoregressive step flow:

    p( next ego step | occupancy patch ahead, goal vector in ego frame, speed )

Rolling it out from a start toward a goal yields a *distribution* of routes.
The re-design simulator: keep start+goal fixed, swap the occupancy patch (move
the furniture) and the sampled routes re-plan around the new layout.  We
validate on LocoReal, where the 4 physical layouts are mutual ground truth:
train on 3 layouts, generate in the held-out layout, and check the generated
routes (a) match the real re-routing (ADE vs a straight-line and a shortest-
collision-free-path baseline) and (b) avoid the real furniture.

Run:  uv run python -m locovrnf.traj              # holdout layout 3 by default
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from locovrnf import config as C
from locovrnf import dataio as D
from locovrnf.model import StepFlow

STEP = 9                # subsample ~90 Hz -> 10 Hz waypoints
STEP_SCALE = 0.5        # ego step units (m) -> ~[-1,1]
AHEAD = 1.0             # look-ahead of the occupancy crop (m)
GOAL_STOP = 0.3         # rollout stops within 0.3 m of goal
MAXSTEPS = 90
SEED = 0


def ego_of(vec, yaw):
    """world (dx,dy) -> ego (lat, fwd) given heading yaw (0=+y toward +x)."""
    sy, cy = np.sin(yaw), np.cos(yaw)
    fwd = vec[..., 0] * sy + vec[..., 1] * cy
    lat = vec[..., 0] * cy - vec[..., 1] * sy
    return np.stack([lat, fwd], -1)


def build_samples(scenes):
    """Per-step training tuples from every LocoReal p1 trajectory in `scenes`."""
    S = {"pos": [], "yaw": [], "goal": [], "scene": []}
    for sid in scenes:
        for t in D.load_scene(sid):
            xy = D.part_xy(t, "p1", "waist")[::STEP]
            if len(xy) < 4:
                continue
            yaw = D.heading_from_motion(xy)
            g = D.goal_xy(t)
            S["pos"].append(xy[:-1]); S["yaw"].append(yaw[:-1])
            S["goal"].append(np.repeat(g[None], len(xy) - 1, 0))
            S["scene"].append(np.full(len(xy) - 1, sid))
            # store next position via a parallel array
            S.setdefault("nxt", []).append(xy[1:])
    return {k: np.concatenate(v).astype(np.float32) if k != "scene"
            else np.concatenate(v) for k, v in S.items()}


def extras_of(pos, yaw, goal):
    """[ego bearing to goal (lat,fwd unit), clipped distance] -> (N,3).
    The caller appends per-step speed to make the 4-dim StepFlow context."""
    gd = goal - pos
    dist = np.linalg.norm(gd, axis=-1, keepdims=True) + 1e-6
    ego_dir = ego_of(gd, yaw) / dist                       # unit bearing (lat,fwd)
    return np.concatenate([ego_dir, np.clip(dist, 0, 4) / 4], -1)


def train(model, S, maps, scenes, dev, rng, epochs, batch):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    split = (rng.random(len(S["pos"])) < 0.15)
    steps_ego = ego_of(S["nxt"] - S["pos"], S["yaw"]) / STEP_SCALE
    speed = np.linalg.norm(S["nxt"] - S["pos"], axis=-1, keepdims=True) / STEP_SCALE
    extra = np.concatenate([extras_of(S["pos"], S["yaw"], S["goal"]), speed], -1).astype(np.float32)

    def run(sel, tr):
        order = rng.permutation(sel); tot = cnt = 0; model.train(tr)
        for sid in scenes:
            m = order[S["scene"][order] == sid]
            for i in range(0, len(m), batch):
                b = m[i:i + batch]
                crop = C.ego_crops(maps[sid], torch.tensor(S["pos"][b], device=dev),
                                   torch.tensor(S["yaw"][b], device=dev), ahead=AHEAD)
                e = torch.tensor(extra[b], device=dev)
                y = torch.tensor(steps_ego[b], device=dev)
                loss = -model.log_prob(crop, e, y).mean()
                if tr:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(b); cnt += len(b)
        return tot / max(cnt, 1)

    tr_sel = np.flatnonzero(~split); va_sel = np.flatnonzero(split)
    for ep in range(epochs):
        tr = run(tr_sel, True)
        with torch.no_grad():
            va = run(va_sel, False)
        print(f"ep{ep} train {tr:.4f} | val {va:.4f}")
    return va


def _field_at(field, pts):
    """Sample a per-pixel field (occupancy or clearance) at world points."""
    px = C.m2px(pts).astype(int)
    col = np.clip(px[..., 0], 0, C.SIZE_PX - 1); row = np.clip(px[..., 1], 0, C.SIZE_PX - 1)
    return field[row, col]


def clearance_px(occ_np):
    """Distance (px) from each free cell to the nearest occupied cell."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(occ_np < 0.5).astype(np.float32)


CLEAR_PX = 10           # required clearance from furniture (~0.1 m)


@torch.no_grad()
def rollout(model, map_xy, occ_np, start, goal, dev, clear_np=None, k=64,
            n_cand=12, temperature=0.7, stall_stop=12):
    """Occupancy-aware rollout: the flow proposes n_cand steps per route, the
    occupancy disposes.  A candidate is accepted only if BOTH its landing and
    the segment midpoint keep clearance from furniture (like a real walker);
    among accepted, goal-progressing candidates we sample weighted by progress
    (preserving multimodality).  k independent routes give the route
    *distribution*.  Returns list of (Ti,2) paths."""
    if clear_np is None:
        clear_np = clearance_px(occ_np)
    pos = np.repeat(np.asarray(start, np.float32)[None], k, 0)
    yaw = np.full(k, float(np.arctan2(goal[0] - start[0], goal[1] - start[1])), np.float32)
    alive = np.ones(k, bool); stall = np.zeros(k, int)
    paths = [[p.copy()] for p in pos]
    rng = np.random.default_rng(SEED)
    for _ in range(MAXSTEPS):
        gd = goal - pos
        dist = np.linalg.norm(gd, axis=-1, keepdims=True) + 1e-6
        ego_dir = ego_of(gd, yaw) / dist
        speed = np.full((k, 1), 0.28 / STEP_SCALE, np.float32)
        extra = np.concatenate([ego_dir, np.clip(dist, 0, 4) / 4, speed], -1).astype(np.float32)
        crop = C.ego_crops(map_xy, torch.tensor(pos, device=dev),
                           torch.tensor(yaw, device=dev), ahead=AHEAD)
        cand = model.sample(crop, torch.tensor(extra, device=dev),
                            n_cand).cpu().numpy() * STEP_SCALE * temperature   # (n_cand,k,2)
        sy, cy = np.sin(yaw), np.cos(yaw)
        # ego (lat,fwd)->world for every candidate
        dx = cand[..., 0] * cy + cand[..., 1] * sy
        dy = -cand[..., 0] * sy + cand[..., 1] * cy
        land = pos[None] + np.stack([dx, dy], -1)                    # (n_cand,k,2)
        mid = pos[None] + 0.5 * np.stack([dx, dy], -1)
        clr = np.minimum(_field_at(clear_np, land), _field_at(clear_np, mid))  # (n_cand,k)
        prog = dist[None, :, 0] - np.linalg.norm(land - goal, axis=-1)  # (n_cand,k) >0 = closer
        for j in range(k):
            if not alive[j]:
                continue
            ok = clr[:, j] >= CLEAR_PX                       # keeps a walker's margin
            good = ok & (prog[:, j] > 0)
            if good.any():
                gi = np.flatnonzero(good)
                w = np.exp(prog[gi, j] / 0.1)                # favour goal-progress, keep spread
                c = gi[rng.choice(len(gi), p=w / w.sum())]
            elif ok.any():
                c = rng.choice(np.flatnonzero(ok))
            else:
                c = int(np.argmax(clr[:, j]))               # least-close-to-furniture fallback
            newp = land[c, j]
            mv = newp - pos[j]
            if np.linalg.norm(mv) > 1e-3:
                yaw[j] = np.arctan2(mv[0], mv[1])
            stall[j] = stall[j] + 1 if prog[c, j] < 0.02 else 0
            pos[j] = newp
            paths[j].append(pos[j].copy())
            if np.linalg.norm(pos[j] - goal) < GOAL_STOP or stall[j] >= stall_stop:
                alive[j] = False
        if not alive.any():
            break
    return [np.array(p, np.float32) for p in paths]


@torch.no_grad()
def score_path(model, map_xy, path, goal, dev):
    """Mean per-step logp the model assigns to an arbitrary polyline route.
    Rollout-independent: directly asks whether the *real* human route is more
    likely under the conditional density than naive alternatives."""
    p = resample(path, 40).astype(np.float32)
    yaw = D.heading_from_motion(p)
    step = (ego_of(p[1:] - p[:-1], yaw[:-1]) / STEP_SCALE).astype(np.float32)
    extra = np.concatenate([extras_of(p[:-1], yaw[:-1], goal),
                            np.linalg.norm(p[1:] - p[:-1], axis=-1, keepdims=True) / STEP_SCALE],
                           -1).astype(np.float32)
    crop = C.ego_crops(map_xy, torch.tensor(p[:-1], device=dev),
                       torch.tensor(yaw[:-1], device=dev), ahead=AHEAD)
    lp = model.log_prob(crop, torch.tensor(extra, device=dev),
                        torch.tensor(step, device=dev))
    return float(lp.mean())


def resample(path, n=20):
    """Arc-length resample a polyline to n points."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return np.repeat(path[:1], n, 0)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, path[:, 0]), np.interp(u, d, path[:, 1])], -1)


def occ_overlap(path, occ):
    px = C.m2px(path).astype(int)
    col = np.clip(px[:, 0], 0, C.SIZE_PX - 1); row = np.clip(px[:, 1], 0, C.SIZE_PX - 1)
    return float(occ[row, col].mean())


def astar_path(occ, start, goal, ds=8):
    """Shortest collision-free path on a downsampled grid (Dijkstra)."""
    small = occ[::ds, ::ds]
    H, W = small.shape
    free = small < 0.5
    idx = -np.ones((H, W), int); idx[free] = np.arange(free.sum())
    rows, cols, data = [], [], []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        rr, cc = np.where(free)
        nr, nc = rr + dr, cc + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        rr, cc, nr, nc = rr[ok], cc[ok], nr[ok], nc[ok]
        ok2 = free[nr, nc]
        rr, cc, nr, nc = rr[ok2], cc[ok2], nr[ok2], nc[ok2]
        rows += idx[rr, cc].tolist(); cols += idx[nr, nc].tolist()
        data += (np.hypot(dr, dc) * np.ones(len(rr))).tolist()
    g = csr_matrix((data, (rows, cols)), shape=(free.sum(), free.sum()))

    def to_small(p):
        px = C.m2px(np.asarray(p)[None])[0]
        return int(np.clip(px[1] // ds, 0, H - 1)), int(np.clip(px[0] // ds, 0, W - 1))

    def nearest_free(rc):
        r, c = rc
        if free[r, c]:
            return r, c
        fr, fc = np.where(free)
        j = np.argmin((fr - r) ** 2 + (fc - c) ** 2)
        return fr[j], fc[j]

    s = nearest_free(to_small(start)); t = nearest_free(to_small(goal))
    dmat, pred = dijkstra(g, indices=idx[s], return_predecessors=True)
    ti = idx[t]
    if not np.isfinite(dmat[ti]):
        return np.array([start, goal], np.float32)
    chain = [ti]
    while chain[-1] != idx[s] and chain[-1] != -9999:
        chain.append(pred[chain[-1]])
    fr, fc = np.where(free)
    pts = [(fc[k] * ds, fr[k] * ds) for k in chain[::-1]]        # (col,row) px
    m = np.stack([np.array(pts)[:, 0] * C.M_PER_PX + C.MAP_MIN,
                  np.array(pts)[:, 1] * C.M_PER_PX + C.MAP_MIN], -1)
    return m.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=3, help="held-out LocoReal layout")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--k", type=int, default=64)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    all_scenes = D.scene_ids_locoreal()
    train_scenes = [s for s in all_scenes if s != args.holdout]
    occs = {s: C.load_occupancy(s) for s in all_scenes}
    maps = {s: C.occ_to_map_tensor(occs[s], dev) for s in all_scenes}
    clears = {s: clearance_px(occs[s]) for s in all_scenes}

    S = build_samples(train_scenes)
    print(f"train steps {len(S['pos'])} from layouts {train_scenes}; holdout {args.holdout}")
    model = StepFlow().to(dev)
    t0 = time.time()
    val = train(model, S, maps, train_scenes, dev, rng, args.epochs, args.batch)
    print(f"trained in {time.time()-t0:.0f}s (val NLL {val:.3f})")

    # ---- evaluate on held-out layout ----
    ho = args.holdout
    real = D.load_scene(ho)
    ade_nf, ade_line, ade_astar = [], [], []
    ov_nf, ov_line, ov_real = [], [], []
    lp_real, lp_line, lp_astar = [], [], []
    for t in real[:60]:
        xy = D.part_xy(t, "p1", "waist")[::STEP]
        if len(xy) < 6:
            continue
        start, goal = xy[0], D.goal_xy(t)
        if np.linalg.norm(goal - start) < 0.5:
            continue
        real_rs = resample(xy)
        routes = rollout(model, maps[ho], occs[ho], start, goal, dev,
                         clear_np=clears[ho], k=args.k)
        gens = np.stack([resample(r) for r in routes])          # (k,20,2)
        ade_k = np.linalg.norm(gens - real_rs[None], axis=-1).mean(-1)
        ade_nf.append(ade_k.min())                              # minADE_k
        line = resample(np.stack([start, goal]))
        ade_line.append(np.linalg.norm(line - real_rs, axis=-1).mean())
        astar_route = astar_path(occs[ho], start, goal)
        ap_ = resample(astar_route)
        ade_astar.append(np.linalg.norm(ap_ - real_rs, axis=-1).mean())
        ov_nf.append(np.mean([occ_overlap(r, occs[ho]) for r in routes]))
        ov_line.append(occ_overlap(resample(np.stack([start, goal]), 60), occs[ho]))
        ov_real.append(occ_overlap(xy, occs[ho]))               # real human, same coarse map
        # rollout-independent likelihood ranking of whole routes
        lp_real.append(score_path(model, maps[ho], xy, goal, dev))
        lp_line.append(score_path(model, maps[ho], np.stack([start, goal]), goal, dev))
        lp_astar.append(score_path(model, maps[ho], astar_route, goal, dev))

    res = {
        "holdout_layout": ho, "n_eval": len(ade_nf),
        "minADE_nf_m": round(float(np.mean(ade_nf)), 3),
        "ADE_straightline_m": round(float(np.mean(ade_line)), 3),
        "ADE_astar_m": round(float(np.mean(ade_astar)), 3),
        "path_occ_overlap_nf": round(float(np.mean(ov_nf)), 4),
        "path_occ_overlap_straightline": round(float(np.mean(ov_line)), 4),
        "path_occ_overlap_real": round(float(np.mean(ov_real)), 4),
        "logp_real_route": round(float(np.mean(lp_real)), 3),
        "logp_straightline_route": round(float(np.mean(lp_line)), 3),
        "logp_astar_route": round(float(np.mean(lp_astar)), 3),
        "real_route_most_likely_frac": round(float(np.mean(
            [r > l and r > a for r, l, a in zip(lp_real, lp_line, lp_astar)])), 3),
        "val_nll": round(float(val), 4),
    }
    # Primary claim: the generated route DISTRIBUTION predicts the real re-routing
    # better than both a naive straight line and a shortest collision-free path.
    res["gate_beats_straightline_ade"] = bool(res["minADE_nf_m"] < res["ADE_straightline_m"])
    res["gate_beats_astar_ade"] = bool(res["minADE_nf_m"] < res["ADE_astar_m"])
    # Secondary: generated routes cut furniture overlap far below a naive straight line
    # (they still clip near the goal approach more than real humans -- honest caveat).
    res["gate_avoids_furniture"] = bool(
        res["path_occ_overlap_nf"] < 0.5 * res["path_occ_overlap_straightline"])
    # Honest negative side-finding (NOT a gate): per-step logp rewards greedy goal
    # pursuit, so a straight line scores higher per step than the real detour route
    # -- local likelihood != global routing quality (the A1/A2 lesson, again).
    res["note_stepwise_logp_favours_straightline"] = bool(
        res["logp_straightline_route"] > res["logp_real_route"])
    res["gate_pass"] = bool(res["gate_beats_astar_ade"] and res["gate_avoids_furniture"])
    C.save_json(res, C.RESULTS / "c3_traj.json")
    print(res)

    # ---- re-design counterfactual figure ----
    make_figures(model, real, occs, maps, clears, ho, all_scenes, dev, args.k, rng)
    print("C3 gate_pass:", res["gate_pass"])


def make_figures(model, real, occs, maps, clears, ho, all_scenes, dev, k, rng):
    # pick a trajectory with a long, non-trivial path
    cand = sorted(real, key=lambda t: -np.linalg.norm(
        D.goal_xy(t) - D.part_xy(t, "p1", "waist")[0]))[:8]
    t = cand[0]
    xy = D.part_xy(t, "p1", "waist")[::STEP]
    start, goal = xy[0], D.goal_xy(t)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # panel 1: generated route distribution vs real, held-out layout
    routes = rollout(model, maps[ho], occs[ho], start, goal, dev, clear_np=clears[ho], k=k)
    ax[0].imshow(occs[ho], cmap="gray_r")
    for r in routes:
        px = C.m2px(r); ax[0].plot(px[:, 0], px[:, 1], "-", color="#c2410c", alpha=.12, lw=1)
    pr = C.m2px(xy); ax[0].plot(pr[:, 0], pr[:, 1], "-", color="#2563eb", lw=2.5, label="real")
    ps, pg = C.m2px(start[None])[0], C.m2px(goal[None])[0]
    ax[0].plot(*ps, "o", color="green", ms=9); ax[0].plot(*pg, "*", color="red", ms=16)
    ax[0].set_title(f"held-out layout {ho}: {k} generated routes (orange)\nvs real (blue)")
    ax[0].legend(loc="lower right"); ax[0].set_xticks([]); ax[0].set_yticks([])

    # panels 2-3: SAME start/goal, re-designed layout (furniture moved) -> reroute
    other = [s for s in all_scenes if s != ho][:2]
    for a, sid in zip(ax[1:], other):
        rr = rollout(model, maps[sid], occs[sid], start, goal, dev, clear_np=clears[sid], k=k)
        a.imshow(occs[sid], cmap="gray_r")
        for r in rr:
            px = C.m2px(r); a.plot(px[:, 0], px[:, 1], "-", color="#7c3aed", alpha=.12, lw=1)
        a.plot(*ps, "o", color="green", ms=9); a.plot(*pg, "*", color="red", ms=16)
        ov = np.mean([occ_overlap(r, occs[sid]) for r in rr])
        a.set_title(f"re-design: same start/goal, layout {sid} furniture\n"
                    f"routes re-plan (occ-overlap {ov:.3f})")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("C3 · goal-conditioned route distribution as a re-design simulator", fontsize=12)
    fig.tight_layout()
    out = C.RESULTS / "c3_redesign.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
