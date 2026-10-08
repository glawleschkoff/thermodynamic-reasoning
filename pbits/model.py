"""Model: 16 p-bits (2 bits per variable), double-well potential, coupling matrix J, clamping plan.

Bit assignment (two bits per variable):
  o0=[0,1]  s0=[2,3]  a0=[4,5]  o1=[6,7]  s1=[8,9]  a1=[10,11]  o2=[12,13]  s2=[14,15]
"""
import jax
import jax.numpy as jnp
import numpy as np

N_BITS = 16

# --- Physics ---
alpha = 4.0
lam = 4.0
beta = 1.2
dt = 0.05   # larger time step: faster relaxation (stable up to ~0.05, unstable from 0.08)
diffusion_std = jnp.sqrt(2.0 * dt / beta)
steps = 10000

bias = jnp.array(np.zeros(N_BITS))

# --- Clamping plan ---
# Bits clamped from the start (currently none)
CLAMPED = {}
# Late-clamped bits: o2 -> square C (-1/-1), o0 -> square B (+1/+1), o1 -> square A (+1/-1)
CLAMPED_LATE = {12: -1.0, 13: -1.0, 0: 1.0, 1: 1.0, 6: 1.0, 7: -1.0}
# Start step per bit: o2 at 200, o0 at 500, o1 at 1180
LATE_START = {12: 200, 13: 200, 0: 500, 1: 500, 6: 1180, 7: 1180}
clamp_start = steps // 3   # only kept for the export (compatibility)

clamp_mask = np.zeros(N_BITS, dtype=bool)
late_mask = np.zeros(N_BITS, dtype=bool)
clamp_val = np.zeros(N_BITS)
for k, v in CLAMPED.items():
    clamp_mask[k] = True
    clamp_val[k] = v
late_start = np.full(N_BITS, steps + 1)
for k, v in CLAMPED_LATE.items():
    late_mask[k] = True
    clamp_val[k] = v
    late_start[k] = LATE_START[k]
late_start = jnp.array(late_start)
clamp_mask = jnp.array(clamp_mask)
late_mask = jnp.array(late_mask)
clamp_val = jnp.array(clamp_val)

# --- Couplings ---
# recoil ~ half the lever, so the undirected model matches the directed POMDP
INERTIA, ACT, RECOIL = 0.9, 0.95, 0.475
c_obs = 1.25   # calibrated: equilibrium matches the exact Bayes posterior (notebooks/test.ipynb) to ~3 %


def build_J():
    J = np.zeros((N_BITS, N_BITS))
    # 1. Sensor model o <-> s
    J[0, 2] = c_obs; J[1, 3] = c_obs
    J[6, 8] = c_obs; J[7, 9] = c_obs
    J[12, 14] = c_obs; J[13, 15] = c_obs
    # 2. Transition t=0 -> t=1: s_t [2,3], a_t [4,5], s_next [8,9]
    J[2, 8] = INERTIA; J[3, 9] = INERTIA          # inertia
    J[4, 8] = -ACT; J[5, 8] = -ACT                # lever of the action on s_next
    J[4, 9] = -ACT; J[5, 9] = ACT
    J[4, 2] = RECOIL; J[5, 2] = RECOIL            # recoil on s_t
    J[4, 3] = RECOIL; J[5, 3] = -RECOIL
    # 3. Transition t=1 -> t=2: s_t [8,9], a_t [10,11], s_next [14,15]
    J[8, 14] = INERTIA; J[9, 15] = INERTIA
    J[10, 14] = -ACT; J[11, 14] = -ACT
    J[10, 15] = -ACT; J[11, 15] = ACT
    J[10, 8] = RECOIL; J[11, 8] = RECOIL
    J[10, 9] = RECOIL; J[11, 9] = -RECOIL
    return jnp.array(J + J.T)


J = build_J()


def V(x):
    """Total energy: double wells + bias + pairwise coupling."""
    return jnp.sum((alpha / 4.0) * (x**4) - (lam / 2.0) * (x**2) - bias * x) - 0.5 * jnp.dot(x, jnp.dot(J, x))


grad_V = jax.grad(V)
