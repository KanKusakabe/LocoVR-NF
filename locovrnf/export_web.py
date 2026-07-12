"""Export the trained C3 StepFlow to a browser-runnable JSON of weights.

Trains the goal-conditioned step flow on all 4 LocoReal layouts, then dumps
every weight (with MADE masks already folded into the hyper linears) plus the
spline/config constants to webdemo/stepflow.json, and a set of test vectors
(crop, extra, base-noise z, resulting step) to webdemo/testvecs.json so the JS
port can be verified to match PyTorch numerically.

Run:  uv run python -m locovrnf.export_web
"""
from __future__ import annotations

import base64
import json

import numpy as np
import torch

from locovrnf import config as C
from locovrnf import dataio as D
from locovrnf import traj as T
from locovrnf.model import StepFlow

WEB = C.ROOT / "webdemo"
EPOCHS = 18


def b64(arr):
    return base64.b64encode(np.asarray(arr, np.float32).tobytes()).decode()


def lin(m):
    return {"w": b64(m.weight.detach().cpu().numpy()), "b": b64(m.bias.detach().cpu().numpy()),
            "shape": list(m.weight.shape)}


def masked_lin(ml):
    w = (ml.mask * ml.weight).detach().cpu().numpy()
    return {"w": b64(w), "b": b64(ml.bias.detach().cpu().numpy()), "shape": list(w.shape)}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.default_rng(0); torch.manual_seed(0)
    scenes = D.scene_ids_locoreal()
    occs = {s: C.load_occupancy(s) for s in scenes}
    maps = {s: C.occ_to_map_tensor(occs[s], dev) for s in scenes}
    S = T.build_samples(scenes)
    print(f"training demo StepFlow on all {len(scenes)} LocoReal layouts, {len(S['pos'])} steps")
    model = StepFlow().to(dev)
    T.train(model, S, maps, scenes, dev, rng, EPOCHS, 2048)
    model.eval()

    # ---- export weights ----
    cnn = model.cnn.net
    out = {
        "config": {"CROP_PX": C.CROP_PX, "CROP_M": C.CROP_M, "AHEAD": T.AHEAD,
                   "STEP_SCALE": T.STEP_SCALE, "MAP_MIN": C.MAP_MIN, "M_PER_PX": C.M_PER_PX,
                   "bins": 8, "bound": 5.0, "slope": 1e-3, "passes": 2,
                   "speed_nominal": 0.28 / T.STEP_SCALE},
        "cnn": {"conv": [{"w": b64(cnn[i].weight.detach().cpu().numpy()),
                          "b": b64(cnn[i].bias.detach().cpu().numpy()),
                          "shape": list(cnn[i].weight.shape)} for i in (0, 2, 4)],
                "lin": lin(cnn[7])},
        "mlp": [lin(model.mlp[0]), lin(model.mlp[2])],
        "merge": lin(model.merge[0]),
        "transforms": [],
    }
    for tr in model.flow.transform.transforms:
        h = tr.hyper
        out["transforms"].append([masked_lin(h[0]), masked_lin(h[2]), masked_lin(h[4])])
    WEB.mkdir(exist_ok=True)
    (WEB / "stepflow.json").write_text(json.dumps(out))
    print("wrote", WEB / "stepflow.json",
          f"({(WEB/'stepflow.json').stat().st_size/1024:.0f} KB)")

    # ---- test vectors (verify JS == PyTorch) ----
    idx = rng.permutation(len(S["pos"]))[:40]
    tv = []
    for k in idx:
        sid = int(S["scene"][k]); pos = S["pos"][k:k+1]; yaw = S["yaw"][k:k+1]
        goal = S["goal"][k:k+1]
        crop = C.ego_crops(maps[sid], torch.tensor(pos, device=dev),
                           torch.tensor(yaw, device=dev), ahead=T.AHEAD)
        extra = np.concatenate([T.extras_of(pos, yaw, goal),
                                np.full((1, 1), out["config"]["speed_nominal"], np.float32)], -1)
        z = torch.tensor(rng.standard_normal((1, 2)).astype(np.float32), device=dev)
        with torch.no_grad():
            ctx = model.context(crop, torch.tensor(extra, device=dev, dtype=torch.float32))
            step = model.flow(ctx).transform.inv(z).cpu().numpy()[0]
        tv.append({"crop": base64.b64encode(
                       (crop[0].cpu().numpy() > 0.5).astype(np.uint8).tobytes()).decode(),
                   "extra": extra[0].tolist(), "z": z.cpu().numpy()[0].tolist(),
                   "step": step.tolist()})
    (WEB / "testvecs.json").write_text(json.dumps(tv))
    print("wrote", WEB / "testvecs.json", f"({len(tv)} vectors)")


if __name__ == "__main__":
    main()
