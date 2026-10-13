import jax
import jax.numpy as jnp
from . import model

beta = 1.2
dt = 0.05
steps = 10000
diffusion_std = jnp.sqrt(2.0 * dt / beta)

CLAMPS = [
    ((12, 13), (-1.0, -1.0), 10.0),   # o2 = C (goal)
    ((0, 1),   (+1.0, +1.0), 25.0),   # o0 = B (observation o_0)
    ((6, 7),   (+1.0, -1.0), 59.0),   # o1 = A (observation o_1)
]


# Euler-Maruyama integration
@jax.jit
def langevin_step(x, inp):
    key, t = inp
    drift = -model.grad_V(x)
    diffusion = jax.random.normal(key, shape=(model.N_PBITS,)) * diffusion_std
    x_next = x + dt * drift + diffusion
    for pbits, vals, t0 in CLAMPS:   # clamping from outside
        x_next = jnp.where(t >= round(t0 / dt), x_next.at[jnp.array(pbits)].set(jnp.array(vals)), x_next)
    return x_next, x_next


def simulate(key):
    keys = jax.random.split(key, steps)
    _, traj = jax.lax.scan(langevin_step, jnp.zeros(model.N_PBITS), (keys, jnp.arange(steps)))
    return traj
