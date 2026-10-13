"""Writes docs/data.js (parameters, trajectory, ensemble, Bayes reference, self-test) and sets the cache timestamp in index.html."""
import json
import os
import re
import time

import jax.numpy as jnp
import numpy as np

from . import bayes
from . import ensemble
import model
import simulate as sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "docs")


def _clamp_pbits():
    """(start step, (pbit, value)) for every clamped p-bit, in plan order."""
    return [(round(t0 / sim.dt), (pbit, v)) for pbits, vals, t0 in sim.CLAMPS for pbit, v in zip(pbits, vals)]


def build_export(traj, ens_probs, stride):
    test_points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(8, model.N_PBITS))
    test_values = [float(model.V(jnp.array(p))) for p in test_points]
    ens_export = {"%d,%d" % pr: np.round(ens_probs[:, i, :], 3).tolist() for i, pr in enumerate(ensemble.PAIRS)}
    return {
        "params": {
            "alpha": model.alpha, "lam": model.lam, "beta": sim.beta, "dt": sim.dt,
            "bias": np.asarray(model.bias).tolist(),
            "J": np.asarray(model.J).tolist(),
            "stride": stride,
            "clamped": {str(b): v for _, (b, v) in _clamp_pbits()},
            "clamp_start": sim.steps // 3 // stride,   # legacy field, unused by the viewer
            "clamp_starts": {str(b): s // stride for s, (b, _) in _clamp_pbits()},
        },
        "traj": np.round(np.asarray(traj)[::stride], 4).tolist(),
        "bayes3": bayes.BAYES_3,
        "bayes2": bayes.BAYES_2,
        "bayes": bayes.BAYES_1,
        "ens": ens_export,
        "selftest": {"x": test_points.tolist(), "V": test_values},
    }


def write(export):
    os.makedirs(VIEWER, exist_ok=True)
    with open(os.path.join(VIEWER, "data.js"), "w") as f:
        f.write("window.PBIT_DATA = " + json.dumps(export, separators=(",", ":")) + ";\n")
    index_path = os.path.join(VIEWER, "index.html")
    with open(index_path) as f:
        html = f.read()
    html = re.sub(r'src="data\.js(\?v=\d+)?"', 'src="data.js?v=%d"' % int(time.time()), html)
    with open(index_path, "w") as f:
        f.write(html)
