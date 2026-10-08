"""Langevin simulation: single trajectory (the ball in the POMDP) and ensemble (probabilities for the 2D plots)."""
import jax
import jax.numpy as jnp
import numpy as np

from . import model as m

SEED = 1   # chosen so that a0 points left and a1 points down during the bee's flights
EXPORT_STRIDE = 2

ENS_K = 500
ENS_SMOOTH = 5   # moving average over export frames (only within sections between clamp times)
# Bit pairs (first bit, second bit); action pairs: lines ↑ → ↓ ←, otherwise A B C D (as in the viewer)
ENS_PAIRS = [(4, 5), (10, 11), (2, 3), (8, 9), (14, 15), (0, 1), (6, 7), (12, 13)]
ENS_ACTION = {(4, 5), (10, 11)}


@jax.jit
def langevin_step(x, inp):
    key, t = inp
    drift = -m.grad_V(x)
    diffusion = jax.random.normal(key, shape=(m.N_BITS,)) * m.diffusion_std
    x_next = x + m.dt * drift + diffusion
    mask_t = m.clamp_mask | (m.late_mask & (t >= m.late_start))
    x_next = jnp.where(mask_t, m.clamp_val, x_next)
    return x_next, x_next


def quad_index(a, b, is_action):
    if is_action:
        return jnp.where(a < 0, jnp.where(b < 0, 0, 1), jnp.where(b > 0, 2, 3))
    return jnp.where(a > 0, jnp.where(b < 0, 0, 1), jnp.where(b < 0, 2, 3))


def simulate():
    """Returns (traj, ens_probs). Replica 0 of the ensemble is exactly the single trajectory."""
    steps = m.steps
    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, steps)
    init_x = jnp.where(m.clamp_mask, m.clamp_val, jnp.zeros(m.N_BITS))
    _, traj = jax.lax.scan(langevin_step, init_x, (keys, jnp.arange(steps)))

    ens_keys = jax.vmap(lambda s: jax.random.split(jax.random.fold_in(key, s), steps))(jnp.arange(ENS_K))
    ens_keys = ens_keys.at[0].set(keys)
    ens_keys = jnp.swapaxes(ens_keys, 0, 1)   # (steps, K, 2)

    def ens_step(X, inp):
        ks, t = inp
        X_next, _ = jax.vmap(lambda x, k: langevin_step(x, (k, t)))(X, ks)
        rows = []
        for (b0, b1) in ENS_PAIRS:
            q = quad_index(X_next[:, b0], X_next[:, b1], (b0, b1) in ENS_ACTION)
            rows.append(jnp.stack([jnp.mean(q == k) for k in range(4)]))
        return X_next, jnp.stack(rows)

    _, ens_probs = jax.lax.scan(ens_step, jnp.tile(init_x, (ENS_K, 1)), (ens_keys, jnp.arange(steps)))
    ens_probs = np.asarray(ens_probs)[::EXPORT_STRIDE]   # (frames, 8, 4)
    return traj, _smooth(ens_probs)


def _smooth(ens_probs):
    if ENS_SMOOTH <= 1:
        return ens_probs
    pad = ENS_SMOOTH // 2
    ker = np.ones(ENS_SMOOTH) / ENS_SMOOTH
    bounds = sorted({0, ens_probs.shape[0]} | {s // 2 for s in m.LATE_START.values() if 0 < s // 2 < ens_probs.shape[0]})
    out = np.empty_like(ens_probs)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = np.pad(ens_probs[lo:hi], ((pad, pad), (0, 0), (0, 0)), mode="edge")
        out[lo:hi] = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 0, seg)
    return out
