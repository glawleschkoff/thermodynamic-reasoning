"""Simulates the p-bit POMDP and writes docs/data.js.

Usage:  python3 -m pipeline.run   (from the repository root)
"""
import jax

import simulate

from . import ensemble, export

SEED = 1            # chosen so that a0 points left and a1 points down during the bee's flights
EXPORT_STRIDE = 2   # every 2nd step goes into data.js


def main():
    key = jax.random.PRNGKey(SEED)
    traj = simulate.simulate(key)
    probs = ensemble.run(key, EXPORT_STRIDE)
    export.write(export.build_export(traj, probs, EXPORT_STRIDE))
    print("docs/data.js written")


if __name__ == "__main__":
    main()
