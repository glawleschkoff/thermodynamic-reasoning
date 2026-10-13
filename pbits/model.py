import jax
import jax.numpy as jnp
import numpy as np

N_PBITS = 16
alpha = 4.0
lam = 4.0
bias = jnp.array(np.zeros(N_PBITS))

J = np.zeros((N_PBITS, N_PBITS))
# Sensor Model p(o_t | s_t)
#          s_t=A  s_t=B  s_t=C  s_t=D
#   o_t=A  0.868  0.064  0.064  0.005
#   o_t=B  0.064  0.868  0.005  0.064
#   o_t=C  0.064  0.005  0.868  0.064
#   o_t=D  0.005  0.064  0.064  0.868
J[0, 2] = 1.25
J[1, 3] = 1.25
J[6, 8] = 1.25
J[7, 9] = 1.25
J[12, 14] = 1.25
J[13, 15] = 1.25
# Transition Model p(s_t+1 | s_t, a_t)
#            | a_t=UP                      | a_t=RIGHT                   | a_t=DOWN                    | a_t=LEFT
#            | s_t=A  s_t=B  s_t=C  s_t=D  | s_t=A  s_t=B  s_t=C  s_t=D  | s_t=A  s_t=B  s_t=C  s_t=D  | s_t=A  s_t=B  s_t=C  s_t=D
#   s_t+1=A  | 0.877  0.120  0.774  0.106  | 0.106  0.003  0.014  0.000  | 0.106  0.014  0.003  0.000  | 0.877  0.774  0.120  0.106
#   s_t+1=B  | 0.120  0.877  0.106  0.774  | 0.774  0.877  0.106  0.120  | 0.014  0.106  0.000  0.003  | 0.003  0.106  0.000  0.014
#   s_t+1=C  | 0.003  0.000  0.106  0.014  | 0.014  0.000  0.106  0.003  | 0.774  0.106  0.877  0.120  | 0.120  0.106  0.877  0.774
#   s_t+1=D  | 0.000  0.003  0.014  0.106  | 0.106  0.120  0.774  0.877  | 0.106  0.774  0.120  0.877  | 0.000  0.014  0.003  0.106
J[2, 8] = 0.9
J[3, 9] = 0.9
J[4, 8] = -0.95
J[4, 9] = -0.95
J[5, 8] = -0.95
J[5, 9] = +0.95
J[2, 4] = +0.475
J[2, 5] = +0.475
J[3, 4] = +0.475
J[3, 5] = -0.475
J[8, 14] = 0.9
J[9, 15] = 0.9
J[10, 14] = -0.95
J[10, 15] = -0.95
J[11, 14] = -0.95
J[11, 15] = +0.95
J[8, 10] = +0.475
J[8, 11] = +0.475
J[9, 10] = +0.475
J[9, 11] = -0.475
J = jnp.array(J + J.T)


def V(x):
    double_wells = jnp.sum((alpha / 4.0) * x**4 - (lam / 2.0) * x**2)
    bias_term = -jnp.dot(bias, x)
    coupling = -0.5 * jnp.dot(x, jnp.dot(J, x))
    return double_wells + bias_term + coupling


grad_V = jax.grad(V)
