"""X1 - side-by-side comparison of the two multi-layer architectures.

Reads the metrics saved by track B (filter) and track H (coarse), plus the
single-flow C3 baseline, and assembles one comparison table + figure.  The point
is NOT a horse race on minADE: both tracks generate comparable route
distributions.  The point is that each unlocks a capability a single flow cannot:

  Track B  modular overlay  -- a separately-trained environment density composed
                              at inference lowers furniture overlap on an UNSEEN
                              layout without retraining the agent (B2).
  Track H  global routing   -- the coarse flow's likelihood ranks the real detour
                              above a furniture-cutting straight line, which the
                              per-step density cannot (H2).

Run:  uv run --no-project python -m locovrnf.compare
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from locovrnf import config as C


def load(name):
    p = C.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    b = load("b_filter.json"); h = load("h_coarse.json"); c3 = load("c3_traj.json")
    ho = (h or b or {}).get("holdout_layout", 3)
    hm = (h or {}).get("per_fold", {}).get(str(ho)) or (h or {}).get("per_fold", {}).get(ho) \
        or next(iter((h or {}).get("per_fold", {}).values()), {})

    # --- assemble the comparison rows (held-out layout) ---
    rows = []
    if c3:
        rows.append(("single flow + hand controller (C3)", c3.get("minADE_nf_m"),
                     c3.get("path_occ_overlap_nf")))
    if b:
        b0 = b["beta0_no_environment"]; bb = b.get("best") or b0
        rows.append(("Track B · agent only (β=0)", b0["minADE_m"], b0["furniture_overlap"]))
        rows.append((f"Track B · agent × environment (β={b.get('best_beta')})",
                     bb["minADE_m"], bb["furniture_overlap"]))
    if hm:
        rows.append(("Track H · coarse→fine", hm.get("minADE_coarse_fine_m"),
                     hm.get("overlap_coarse_fine")))
        rows.append(("Track H · hand controller (ref)", hm.get("minADE_hand_controller_m"),
                     hm.get("overlap_hand_controller")))
    # naive references
    if hm:
        rows.append(("straight line", hm.get("ADE_straightline_m"), None))

    table = {
        "holdout_layout": ho,
        "rows": [{"method": m, "minADE_m": a, "furniture_overlap": o} for m, a, o in rows],
        "capability_B2_modular_overlay": {
            "claim": ("environment density trained on other layouts, composed at "
                      "inference, lowers overlap on the unseen layout"),
            "overlap_no_env": b and b["beta0_no_environment"]["furniture_overlap"],
            "overlap_with_env": b and (b.get("best") or {}).get("furniture_overlap"),
            "gate_pass": b and b.get("gate_B2_modular_overlay"),
        } if b else None,
        "capability_H2_global_routing": {
            "claim": ("coarse-flow likelihood ranks the real detour above a straight "
                      "line, reversing the per-step density"),
            "coarse_real_higher_frac": hm.get("coarse_real_higher_frac"),
            "stepwise_real_higher_frac": hm.get("stepwise_real_higher_frac"),
            "gate_pass": (h or {}).get("gate_H2_global_discrimination"),
        } if hm else None,
        "summary": ("Both tracks produce comparable route distributions; each adds a "
                    "capability a single flow lacks -- B: inference-time layer swap, "
                    "H: global route discrimination. They are complementary (a coarse "
                    "plan whose fine steps use the product-B environment expert = X2)."),
    }
    C.save_json(table, C.RESULTS / "x_compare.json")

    # --- figure: minADE + overlap bars, and the two capability panels ---
    short = {
        "single flow + hand controller (C3)": "C3\nhand ctrl",
        "Track B · agent only (β=0)": "B: agent\n(β=0)",
        f"Track B · agent × environment (β={b.get('best_beta') if b else ''})": "B: agent×env\n(β=4)",
        "Track H · coarse→fine": "H: coarse\n→fine",
        "Track H · hand controller (ref)": "H: hand\nctrl (ref)",
        "straight line": "straight\nline",
    }
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = [short.get(r[0], r[0]) for r in rows]
    ade = [r[1] if r[1] is not None else np.nan for r in rows]
    ov = [r[2] if r[2] is not None else np.nan for r in rows]
    cols = ["#6b7280", "#9ca3af", "#7c3aed", "#0d9488", "#94a3b8", "#cbd5e1"][:len(rows)]
    x = np.arange(len(rows))
    ax[0].bar(x, ade, color=cols)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8, rotation=0)
    ax[0].set_ylabel("minADE / ADE  [m]  ↓"); ax[0].set_title(f"route error, held-out layout {ho}")
    for i, v in enumerate(ade):
        if not np.isnan(v):
            ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax[1].bar(x, ov, color=cols)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("furniture overlap  [fraction]  ↓"); ax[1].set_title("furniture overlap")
    for i, v in enumerate(ov):
        if not np.isnan(v):
            ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # capability panel (text)
    ax[2].axis("off")
    lines = ["Two capabilities a single flow lacks", ""]
    if b:
        lines += ["Track B · modular overlay (B2)",
                  f"  overlap  no-env {b['beta0_no_environment']['furniture_overlap']:.3f}"
                  f"  →  with-env {(b.get('best') or {}).get('furniture_overlap', float('nan')):.3f}",
                  f"  on the UNSEEN layout, env composed at inference", ""]
    if hm:
        lines += ["Track H · global routing (H2)",
                  f"  per-step logp picks real  {hm.get('stepwise_real_higher_frac')}",
                  f"  coarse   logp picks real  {hm.get('coarse_real_higher_frac')}",
                  f"  (>0.5 = real detour ranked above straight line)"]
    ax[2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10,
               family="monospace", transform=ax[2].transAxes)
    fig.suptitle("X1 · two multi-layer NF architectures on the same space", fontsize=13)
    fig.tight_layout()
    out = C.RESULTS / "x_compare.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out); print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
