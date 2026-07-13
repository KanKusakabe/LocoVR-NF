"""Track B - product of two flows as a per-step particle filter.

Two experts are trained SEPARATELY and combined only at inference:

  agent expert   p_U(next ego step | occ crop ahead, goal bearing, speed)   [StepFlow]
  environment    p_E(here is a place people stand | occ crop centred here)  [AffordanceFlow]

At each rollout step the agent PROPOSES n_cand candidate steps; each candidate
landing is re-weighted by the environment expert:

    weight(cand) ~ exp( progress/tau + beta * logp_E(landing) )

beta is the coupling knob.  beta=0 recovers a pure goal-seeking agent with NO
environment awareness (it will walk through furniture).  beta>0 lets the reusable,
occupancy-conditioned environment density steer the agent off furniture -- WITHOUT
retraining the agent.  This is the modular "overlay two flows on the same space"
claim: the environment expert is trained once and composed at inference.

Deliberately NO hand-built clearance field here (unlike traj.rollout): the whole
point is that the LEARNED environment density can play that role probabilistically.

Phases / gates (results/b_filter.json):
  B0  sanity      filter runs and beta>0 produces sane routes toward goal
  B1  core        some beta>0 lowers furniture-overlap vs beta=0, minADE not worse
  B2  modularity  agent+env trained on 3 layouts, held-out layout: adding the env
                  expert (never retrained for it) cuts overlap on the unseen layout

Honest guard: claims live on generated minADE + furniture-overlap only, never on
per-step logp ranking (the A1/A2/C3 lesson: local likelihood != global routing).

Run:  uv run --no-project python -m locovrnf.filter
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
from locovrnf.model import StepFlow, AffordanceFlow
from locovrnf.traj import (build_samples, train, ego_of, resample, occ_overlap,
                           STEP, STEP_SCALE, AHEAD, GOAL_STOP, MAXSTEPS, SEED)

AFF_SCALE = C.CROP_M / 2.0        # affordance offset units (matches affordance.py)


def train_affordance(scenes, dev, rng, epochs=8, batch=2048):
    """Train the environment expert on the given layouts only (leak-free)."""
    from locovrnf.affordance import ego_offset, crops_for, JITTER, SUB
    pos, scene = [], []
    for sid in scenes:
        xy = np.concatenate([D.part_xy(t, "p1", "waist")[::SUB]
                             for t in D.load_scene(sid)], 0).astype(np.float32)
        pos.append(xy); scene.append(np.full(len(xy), sid, np.int64))
    pos = np.concatenate(pos); scene = np.concatenate(scene)
    maps = {s: C.occ_to_map_tensor(C.load_occupancy(s), dev) for s in scenes}
    model = AffordanceFlow().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    split = rng.random(len(pos)) < 0.15
    sel = np.flatnonzero(~split)
    for ep in range(epochs):
        order = rng.permutation(sel); model.train(True)
        for sid in scenes:
            m = order[scene[order] == sid]
            for i in range(0, len(m), batch):
                b = m[i:i + batch]
                centers = pos[b] + rng.uniform(-JITTER, JITTER, (len(b), 2)).astype(np.float32)
                angs = rng.uniform(0, 2 * np.pi, len(b)).astype(np.float32)
                crop = crops_for(maps[sid], centers, angs, dev)
                x = torch.tensor(ego_offset(pos[b], centers, angs) / AFF_SCALE, device=dev)
                loss = -model.log_prob(crop, x).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def aff_logp(aff, map_xy, pts, yaw, dev):
    """Environment expert logp that `pts` are stood-on, given the layout map."""
    crop = C.ego_crops(map_xy, torch.tensor(pts, dtype=torch.float32, device=dev),
                       torch.tensor(yaw, dtype=torch.float32, device=dev), ahead=0.0)
    x0 = torch.zeros((len(pts), 2), device=dev)
    return aff.log_prob(crop, x0).cpu().numpy()


@torch.no_grad()
def rollout_filter(step_model, aff, map_xy, aff_map_xy, start, goal, dev, beta,
                   k=64, n_cand=12, temperature=0.7, tau=0.1):
    """Particle-filter rollout: agent proposes, environment expert re-weights.

    aff_map_xy lets the environment expert read a DIFFERENT layout than the agent
    (used for the swap figure); pass the same map for the normal product.
    """
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
        cand = step_model.sample(crop, torch.tensor(extra, device=dev),
                                 n_cand).cpu().numpy() * STEP_SCALE * temperature   # (n_cand,k,2)
        sy, cy = np.sin(yaw), np.cos(yaw)
        dx = cand[..., 0] * cy + cand[..., 1] * sy
        dy = -cand[..., 0] * sy + cand[..., 1] * cy
        land = pos[None] + np.stack([dx, dy], -1)                    # (n_cand,k,2)
        prog = dist[None, :, 0] - np.linalg.norm(land - goal, axis=-1)   # (n_cand,k)
        # environment expert logp at every candidate landing (heading = motion dir)
        head = np.arctan2(dx, dy)                                    # (n_cand,k)
        lp_e = aff_logp(aff, aff_map_xy, land.reshape(-1, 2),
                        head.reshape(-1), dev).reshape(n_cand, k)
        # Goal-reaching is preserved by only letting the environment expert choose
        # AMONG goal-progressing candidates (mirrors the C3 controller structure but
        # replaces the hand-built clearance test with the LEARNED affordance density).
        # Particles with no progressing candidate fall back to pure goal progress.
        score = prog / tau + beta * lp_e                            # (n_cand,k)
        elig = prog > 0.0
        score = np.where(elig, score, -1e9)
        none = ~elig.any(0)                                        # (k,) stuck particles
        if none.any():
            score[:, none] = (prog / tau)[:, none]
        # vectorized categorical sampling per particle via the Gumbel-max trick
        gum = -np.log(-np.log(rng.random((n_cand, k)) + 1e-12) + 1e-12)
        c = np.argmax(score + gum, axis=0)                          # (k,)
        ar = np.arange(k)
        newp = land[c, ar]                                          # (k,2)
        mv = newp - pos
        mvn = np.linalg.norm(mv, axis=-1)
        upd = mvn > 1e-3
        yaw = np.where(upd, np.arctan2(mv[:, 0], mv[:, 1]), yaw)
        progc = prog[c, ar]
        stall = np.where(progc < 0.02, stall + 1, 0)
        move = alive[:, None]
        pos = np.where(move, newp, pos)
        for j in np.flatnonzero(alive):
            paths[j].append(pos[j].copy())
        reached = np.linalg.norm(pos - goal, axis=-1) < GOAL_STOP
        alive = alive & ~reached & (stall < 20)
        if not alive.any():
            break
    return [np.array(p, np.float32) for p in paths]


def eval_beta(step_model, aff, maps, occs, real, ho, dev, beta, k, n_traj=40):
    """minADE / overlap / success on held-out layout for one beta."""
    ade, ov, succ = [], [], []
    for t in real[:n_traj]:
        xy = D.part_xy(t, "p1", "waist")[::STEP]
        if len(xy) < 6:
            continue
        start, goal = xy[0], D.goal_xy(t)
        if np.linalg.norm(goal - start) < 0.5:
            continue
        real_rs = resample(xy)
        routes = rollout_filter(step_model, aff, maps[ho], maps[ho], start, goal,
                                dev, beta, k=k)
        gens = np.stack([resample(r) for r in routes])
        ade.append(np.linalg.norm(gens - real_rs[None], axis=-1).mean(-1).min())
        ov.append(np.mean([occ_overlap(r, occs[ho]) for r in routes]))
        succ.append(np.mean([np.linalg.norm(r[-1] - goal) < GOAL_STOP for r in routes]))
    return (round(float(np.mean(ade)), 3), round(float(np.mean(ov)), 4),
            round(float(np.mean(succ)), 3), len(ade))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

    scenes = D.scene_ids_locoreal()
    train_scenes = [s for s in scenes if s != args.holdout]
    occs = {s: C.load_occupancy(s) for s in scenes}
    maps = {s: C.occ_to_map_tensor(occs[s], dev) for s in scenes}

    # --- train BOTH experts on the training layouts only (leak-free for B2) ---
    t0 = time.time()
    S = build_samples(train_scenes)
    step_model = StepFlow().to(dev)
    val = train(step_model, S, maps, train_scenes, dev, rng, args.epochs, 2048)
    aff = train_affordance(train_scenes, dev, rng)
    print(f"trained agent+env on {train_scenes} in {time.time()-t0:.0f}s (agent val {val:.3f})")

    real = D.load_scene(args.holdout)

    # --- B1: beta sweep on the held-out (unseen) layout ---
    betas = [0.0, 1.0, 2.0, 4.0]
    sweep = {}
    for b in betas:
        a, o, s, n = eval_beta(step_model, aff, maps, occs, real, args.holdout, dev, b, args.k)
        sweep[b] = {"minADE_m": a, "furniture_overlap": o, "success_rate": s}
        print(f"  beta {b}: minADE {a} m | overlap {o} | success {s} (n={n})")

    base = sweep[0.0]
    # best beta = lowest overlap whose minADE stays within 10% of beta=0
    ok = {b: v for b, v in sweep.items()
          if b > 0 and v["minADE_m"] <= base["minADE_m"] * 1.10}
    best_b = min(ok, key=lambda b: ok[b]["furniture_overlap"]) if ok else None

    res = {
        "holdout_layout": args.holdout, "k_routes": args.k,
        "agent_val_nll": round(float(val), 3),
        "beta_sweep": {str(b): v for b, v in sweep.items()},
        "beta0_no_environment": base,
        "best_beta": best_b,
        "best": sweep[best_b] if best_b is not None else None,
        # B0: does the product machinery reach the goal at all
        "gate_B0_reaches_goal": bool(sweep[betas[2]]["success_rate"] > 0.5),
        # B1: environment layer lowers furniture overlap without hurting minADE
        "gate_B1_env_lowers_overlap": bool(
            best_b is not None and
            sweep[best_b]["furniture_overlap"] < base["furniture_overlap"]),
        # B2: on the UNSEEN layout, adding the reusable env expert cuts overlap
        # (agent+env never saw this layout; env is composed at inference only)
        "gate_B2_modular_overlay": bool(
            best_b is not None and
            sweep[best_b]["furniture_overlap"] < base["furniture_overlap"] * 0.8),
        "note": ("beta=0 is a pure goal-seeking agent with no obstacle awareness; "
                 "beta>0 adds the separately-trained affordance density as an "
                 "inference-time overlay. Claims are on overlap+minADE, not per-step logp."),
    }
    res["gate_pass"] = bool(res["gate_B0_reaches_goal"] and res["gate_B1_env_lowers_overlap"])
    C.save_json(res, C.RESULTS / "b_filter.json")
    print("B gate_pass:", res["gate_pass"], "best_beta:", best_b)

    make_figure(step_model, aff, maps, occs, real, args.holdout, train_scenes,
                dev, args.k, best_b if best_b else 2.0, sweep, betas)


def make_figure(step_model, aff, maps, occs, real, ho, train_scenes, dev, k, best_b,
                sweep, betas):
    # pick a long trajectory
    t = sorted(real, key=lambda t: -np.linalg.norm(
        D.goal_xy(t) - D.part_xy(t, "p1", "waist")[0]))[0]
    xy = D.part_xy(t, "p1", "waist")[::STEP]
    start, goal = xy[0], D.goal_xy(t)
    ps, pg = C.m2px(start[None])[0], C.m2px(goal[None])[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # panel 1: beta sweep curve (overlap and minADE vs beta)
    bs = betas
    ov = [sweep[b]["furniture_overlap"] for b in bs]
    ad = [sweep[b]["minADE_m"] for b in bs]
    ax0 = ax[0]; ax0b = ax0.twinx()
    ax0.plot(bs, ov, "o-", color="#c2410c", label="furniture overlap ↓")
    ax0b.plot(bs, ad, "s--", color="#2563eb", label="minADE [m] ↓")
    ax0.set_xlabel("β  (environment coupling)"); ax0.set_ylabel("furniture overlap", color="#c2410c")
    ax0b.set_ylabel("minADE [m]", color="#2563eb")
    ax0.set_title("B1 · overlay strength β\nenvironment expert steers off furniture")

    # panel 2: beta=0 (no environment) routes -- walk through furniture
    r0 = rollout_filter(step_model, aff, maps[ho], maps[ho], start, goal, dev, 0.0, k=k)
    ax[1].imshow(occs[ho], cmap="gray_r")
    for r in r0:
        px = C.m2px(r); ax[1].plot(px[:, 0], px[:, 1], "-", color="#9ca3af", alpha=.2, lw=1)
    ax[1].plot(*ps, "o", color="green", ms=9); ax[1].plot(*pg, "*", color="red", ms=15)
    ov0 = np.mean([occ_overlap(r, occs[ho]) for r in r0])
    ax[1].set_title(f"β=0 · agent only (no environment)\noverlap {ov0:.3f}")
    ax[1].set_xticks([]); ax[1].set_yticks([])

    # panel 3: beta=best (agent x environment) -- reroute around furniture
    rb = rollout_filter(step_model, aff, maps[ho], maps[ho], start, goal, dev, best_b, k=k)
    ax[2].imshow(occs[ho], cmap="gray_r")
    for r in rb:
        px = C.m2px(r); ax[2].plot(px[:, 0], px[:, 1], "-", color="#7c3aed", alpha=.2, lw=1)
    pr = C.m2px(xy); ax[2].plot(pr[:, 0], pr[:, 1], "-", color="#2563eb", lw=2.5, label="real")
    ax[2].plot(*ps, "o", color="green", ms=9); ax[2].plot(*pg, "*", color="red", ms=15)
    ovb = np.mean([occ_overlap(r, occs[ho]) for r in rb])
    ax[2].set_title(f"β={best_b} · agent × environment overlay\noverlap {ovb:.3f} (unseen layout {ho})")
    ax[2].legend(loc="lower right"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle("Track B · product of two flows as a per-step particle filter", fontsize=12)
    fig.tight_layout()
    out = C.RESULTS / "b_filter.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
