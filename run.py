"""Simulates the p-bit POMDP and writes docs/data.js.

Usage:  python3 run.py
"""
from pbits import export, simulate


def main():
    traj, ens_probs = simulate.simulate()
    export.write(export.build_export(traj, ens_probs))
    print("docs/data.js written")


if __name__ == "__main__":
    main()
