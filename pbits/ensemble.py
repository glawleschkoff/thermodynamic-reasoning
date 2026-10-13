"""Ensemble of independent replicas: the ergodicity check.

The ball in the POMDP is a single trajectory. Here K replicas of the same dynamics (same model, same clamping) are
simulated in parallel. The fraction of replicas per quadrant at time t is the distribution p(state, t) that the single
ball samples from. These are the probabilities shown in the 2D plots.

Replica 0 uses the same random keys as the single trajectory and is therefore exactly that trajectory.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import model
from . import simulate as sim

K = 500
SMOOTH = 5   # moving average over export frames (only within sections between clamp times)
# p-bit pairs (first p-bit, second p-bit); action pairs: lines up/right/down/left, otherwise A B C D (as in the viewer)
PAIRS = [(4, 5), (10, 11), (2, 3), (8, 9), (14, 15), (0, 1), (6, 7), (12, 13)]
ACTION = {(4, 5), (10, 11)}


def quad_index(a, b, is_action):
    if is_action:
        return jnp.where(a < 0, jnp.where(b < 0, 0, 1), jnp.where(b > 0, 2, 3))
    return jnp.where(a > 0, jnp.where(b < 0, 0, 1), jnp.where(b < 0, 2, 3))


def run(key, stride):
    """Quadrant probabilities per export frame, shape (frames, len(PAIRS), 4)."""
    steps = sim.steps
    keys = jax.random.split(key, steps)
    ens_keys = jax.vmap(lambda s: jax.random.split(jax.random.fold_in(key, s), steps))(jnp.arange(K))
    ens_keys = ens_keys.at[0].set(keys)
    ens_keys = jnp.swapaxes(ens_keys, 0, 1)   # (steps, K, 2)

    def step(X, inp):
        ks, t = inp
        X_next, _ = jax.vmap(lambda x, k: sim.langevin_step(x, (k, t)))(X, ks)
        rows = []
        for (b0, b1) in PAIRS:
            q = quad_index(X_next[:, b0], X_next[:, b1], (b0, b1) in ACTION)
            rows.append(jnp.stack([jnp.mean(q == k) for k in range(4)]))
        return X_next, jnp.stack(rows)

    _, probs = jax.lax.scan(step, jnp.zeros((K, model.N_PBITS)), (ens_keys, jnp.arange(steps)))
    return _smooth(np.asarray(probs)[::stride])


def _smooth(probs):
    if SMOOTH <= 1:
        return probs
    pad = SMOOTH // 2
    ker = np.ones(SMOOTH) / SMOOTH
    starts = (round(t0 / sim.dt) // 2 for _, _, t0 in sim.CLAMPS)
    bounds = sorted({0, probs.shape[0]} | {s for s in starts if 0 < s < probs.shape[0]})
    out = np.empty_like(probs)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = np.pad(probs[lo:hi], ((pad, pad), (0, 0), (0, 0)), mode="edge")
        out[lo:hi] = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 0, seg)
    return out
