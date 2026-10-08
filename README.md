# Thermodynamic Reasoning – p-bit POMDP

Sixteen p-bits (Langevin dynamics in a double-well potential) encode a POMDP with three hidden
states, three observations and two actions (two p-bits = four states per variable). A 3D viewer
plays back the precomputed simulation together with 2D probability plots and the exact Bayesian
posteriors as a reference.

## Structure
- `pbits/model.py` – parameters, coupling matrix J, clamping plan
- `pbits/simulate.py` – Langevin simulation (JAX) and the ensemble used for the probabilities
- `pbits/bayes.py` – exact Bayes posteriors (reference, derived in `notebooks/test.ipynb`)
- `pbits/export.py` – writes `viewer/data.js`
- `run.py` – run the simulation and export the data
- `dev.py` – local server (port 8000); re-runs `run.py` when the Python files change
- `docs/` – browser viewer (three.js), also served by GitHub Pages; `docs/vendor` and `docs/assets` are stored locally
- `archive/` – older variants and iterations (not part of the repository)

## Usage
    pip install -r requirements.txt
    python3 run.py      # simulation -> docs/data.js
    python3 dev.py      # http://localhost:8000

## Demo
The viewer is static. On GitHub Pages choose "Deploy from a branch", branch `main`, folder `/docs`.
`docs/data.js` is part of the repository, so no Python is needed to view the demo.

## Credits and licenses
- Code: MIT, see `LICENSE`.
- Bee model: "Cube Pets" by Kenney (kenney.nl), CC0.
- Honeycomb model: "Honeycomb" by Poly by Google, licensed under CC BY 3.0
  (https://creativecommons.org/licenses/by/3.0/). Attribution: Poly by Google. The model was converted/used
  unchanged except for scaling and placement. Please add the exact author name and source link from the
  download page here.
- three.js (MIT), included in `docs/vendor`.
