"""Writes docs/data.js (parameters, trajectory, ensemble, Bayes reference, self-test) and sets the cache timestamp in index.html."""
import json
import os
import re
import time

import jax.numpy as jnp
import numpy as np

from . import bayes
from . import model as m
from . import simulate as sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "docs")


def build_export(traj, ens_probs):
    stride = sim.EXPORT_STRIDE
    test_points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(8, m.N_BITS))
    test_values = [float(m.V(jnp.array(p))) for p in test_points]
    ens_export = {"%d,%d" % pr: np.round(ens_probs[:, i, :], 3).tolist() for i, pr in enumerate(sim.ENS_PAIRS)}
    return {
        "params": {
            "alpha": m.alpha, "lam": m.lam, "beta": m.beta, "dt": m.dt,
            "bias": np.asarray(m.bias).tolist(),
            "J": np.asarray(m.J).tolist(),
            "stride": stride,
            "clamped": {str(k): v for k, v in {**m.CLAMPED, **m.CLAMPED_LATE}.items()},
            "clamp_start": m.clamp_start // stride,
            "clamp_starts": {str(k): s // stride for k, s in m.LATE_START.items()},
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
