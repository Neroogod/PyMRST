# PyRevSimAD-Inspired Tensor and Torch Backend Notes

This document records the additions that came from looking at
`/Users/Shared/PyMRST_workspace/PyRevSimAD` and the newer
PyTorch backend work in PyMRST. The goal is not to replace the MRST-parity
NumPy/SciPy/ADI implementation. The goal is to add a second path for tensor
exports, ML datasets, GPU experiments, and Torch-autograd residuals while
keeping MRST validation intact.

## What Came From PyRevSimAD

PyRevSimAD had a useful idea: reservoir grids and simulation states should be
easy to move into tensor form for GPU, autograd, and ML workflows. PyMRST now
has the beginning of that workflow, but with a stricter precision policy for
MRST parity.

The main additions are:

- MRST `.mat` grid import utility
- grid/rock/state tensor exporters
- saved-state dataset wrappers
- optional Torch backend infrastructure
- backend TPFA, oil-water, and black-oil prototype solvers
- SPE1 Torch backend runner and validation

## Precision Policy

PyRevSimAD used `float32` in places because it was oriented toward ML/GPU work.
PyMRST defaults to `float64` because the baseline target is MRST numerical
parity.

Use `float32` only when you explicitly want ML speed or lower memory use:

```python
from pymrst.ml import grid_to_tensors

tensors64 = grid_to_tensors(G, backend="torch", device="cpu")
tensors32 = grid_to_tensors(G, backend="torch", device="cuda", dtype="float32")
```

## MRST Grid Import

PyMRST now has a utility for loading MRST grid structs saved to `.mat` files:

```python
from pymrst.gridprocessing.load_mrst_grid_mat import load_mrst_grid_mat

G = load_mrst_grid_mat("grid_from_mrst.mat")
```

This is useful when a MATLAB/MRST script has already built a grid and you want
to inspect, plot, export, or reuse it from Python.

## Full MRST `.mat` Bridge

PyMRST now also has a broader MATLAB/MRST result bridge under `pymrst.io`.
It normalizes MATLAB structs/cell arrays into Python dictionaries/lists and
converts known MRST objects into PyMRST-native objects where the mapping is
safe.

MATLAB side:

```matlab
save('mrst_case.mat', 'G', 'states', 'wellSols', 'schedule', 'W');
```

Python side:

```python
from pymrst.io import (
    load_mrst_mat,
    load_mrst_grid_mat,
    load_mrst_states_mat,
    load_mrst_well_sols_mat,
    load_mrst_schedule_mat,
    load_mrst_wells_mat,
)

raw = load_mrst_mat("mrst_case.mat")
G = load_mrst_grid_mat("mrst_case.mat")
states = load_mrst_states_mat("mrst_case.mat")
well_sols = load_mrst_well_sols_mat("mrst_case.mat")
schedule = load_mrst_schedule_mat("mrst_case.mat")
W = load_mrst_wells_mat("mrst_case.mat")
```

Known conversions:

- `G` -> PyMRST `Grid`, with MRST one-based indices converted to zero-based.
- `states` -> list of PyMRST `State` dictionaries.
- `wellSols` -> list of timestep well-solution dictionaries.
- `W` -> list of PyMRST `Well` objects when the standard MRST well fields are
  present.
- `schedule` -> normalized dictionary, with converted control wells stored as
  `control[i]["W_py"]` while the raw `W` fields are preserved.

Unknown fields are intentionally kept as raw normalized dictionaries/arrays so
the import does not throw away MRST data PyMRST does not understand yet.

## Tensor Export Utilities

The tensor export functions live in `pymrst.ml`:

```python
from pymrst.ml import grid_to_tensors, rock_to_tensors, state_to_tensors

grid_t = grid_to_tensors(G, backend="torch", device="cuda")
rock_t = rock_to_tensors(rock, backend="torch", device="cuda")
state_t = state_to_tensors(result.states[-1], backend="torch", device="cuda")
```

Available helpers:

- `grid_to_tensors(G, backend="numpy" | "torch", device=None, dtype="float64")`
- `rock_to_tensors(rock, backend="numpy" | "torch", device=None, dtype="float64")`
- `state_to_tensors(state, backend="numpy" | "torch", device=None, dtype="float64")`

These are for ML/surrogate/GNN workflows. They do not change the validated ADI
solver path.

## Saved Simulation Dataset Format

The SPE1 tutorial can save states plus grid/rock/well metadata:

```bash
cd /Users/Shared/PyMRST_workspace/PyMRST
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --steps 10 \
  --save-states examples/results_spe1/spe1_states.npz
```

You can load the saved states as an ML dataset:

```python
from pymrst.ml import SPE1Dataset

dataset = SPE1Dataset(
    "examples/results_spe1/spe1_states.npz",
    backend="torch",
    device="cuda",
)

x, y = dataset[0]  # x = step 0 fields, y = step 1 fields
```

The dataset wrapper is intended for sequence learning, surrogate modeling, and
state-to-state prediction experiments.

## PyVista and Plotting

PyMRST also gained stronger plotting support around saved states and SPE1:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --steps 10 \
  --plot \
  --save-states examples/results_spe1/spe1_states.npz
```

For interactive state browsing:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --steps 10 \
  --toolbar
```

For SPE1, remember the grid is physically thin in `z`, so PyVista may look
flattened unless you use a vertical exaggeration/`zscale` option in the plotting
helpers.

## Browser-Based Interactive Plotting

Inspired by the WGLMakie/JutulDarcy browser workflow, PyMRST now has an
optional browser viewer based on Plotly/WebGL. It writes a standalone HTML file
that can be opened in any browser, with 3D rotate/zoom, a field/timestep
dropdown, and legend toggles for wells and grid lines.

Install the optional dependency:

```bash
cd /Users/Shared/PyMRST_workspace/PyMRST
pip install -e '.[browser]'
```

Create a browser viewer from saved SPE1 states:

```bash
python3 -m pymrst.plotting.extensions.plot_saved_states \
  examples/results_spe1/spe1_states.npz \
  --browser \
  --field sG \
  --zscale 5 \
  --output examples/results_spe1/spe1_browser.html
```

Open it automatically after writing:

```bash
python3 -m pymrst.plotting.extensions.plot_saved_states \
  examples/results_spe1/spe1_states.npz \
  --browser \
  --field sG \
  --open-browser
```

From Python:

```python
from pymrst.plotting import plot_reservoir_browser

plot_reservoir_browser(
    G,
    {"pressure": pressure_history, "sG": gas_saturation_history},
    wells=W,
    times=report_times,
    filename="reservoir.html",
    zscale=5,
)
```

Current browser plotting scope:

- triangulated reservoir cell-body rendering with `Mesh3d`
- timestep slider and play/pause controls
- grid-line trace
- well traces
- browser-native rotate/zoom/pan
- standalone HTML output

For very large grids, use `--browser-style points` to fall back to the lighter
cell-centroid viewer. Use `--browser-all-fields` only for small cases; for SPE1,
plot one field at a time, for example `--field sG` or `--field pressure`.

This is intentionally separate from `plot_toolbar`: `plot_toolbar` is the
desktop PyVista/Qt inspector, while `plot_reservoir_browser` is the browser
viewer.

## Optional Array Backend

The backend layer is in `pymrst.backends`.

Core entry point:

```python
from pymrst.backends import get_backend

be = get_backend("torch", device="cuda", dtype="float64")
x = be.asarray([1.0, 2.0, 3.0])
```

Supported backend names:

- `numpy`
- `torch`

Solver modes currently used by backend paths:

- dense Torch solve
- SciPy sparse solve
- optional CuPy sparse solve when CuPy is installed
- matrix-free Torch GMRES through JVPs

## Backend TPFA

The incompressible backend TPFA path validates pressure, fluxes, simple wells,
sources, and gravity against PyMRST's CPU solver:

```bash
python3 examples/validation/validate_backend_tpfa.py
```

This is the simplest backend pressure-solve slice and is useful for checking
Torch/CUDA availability.

## Backend Oil-Water

There is a small fully implicit oil-water Torch-autograd backend:

```bash
python3 examples/validation/validate_backend_oil_water.py
```

This validates:

- residual assembly
- Torch autograd Jacobian
- backend Newton stepping

It is a research/backend path, not a replacement for the MRST-parity ADI model.

## Backend Black-Oil Fluid Tables

SPE1 black-oil PVT/relative-permeability tables now have backend-aware Torch
interpolation:

```bash
python3 examples/validation/validate_backend_blackoil_fluid.py
```

Validated properties include:

- `bW`, `muW`, `pvMult`
- `bG`, `muG`
- `rsSat`
- `bO`, `muO`
- `krW`, `krG`, `krO`

## Backend Live-Oil Black-Oil Residual

The Torch backend has a live-oil black-oil residual matching the existing SPE1
ADI equations:

```bash
python3 examples/validation/validate_backend_blackoil_residual.py
```

It supports:

- pressure primary variable
- water saturation primary variable
- switched hydrocarbon primary variable: `Rs` in undersaturated cells, `sG` in
  saturated cells
- backend PVT table interpolation
- three-phase wells
- dissolved-gas well accounting
- Torch autograd through the residual

## Backend Black-Oil Model

`BackendBlackOilModel` adds a Torch Newton loop around the residual. It performs
black-oil flash/status updates after every Newton increment, matching the
MRST/PyMRST ADI control flow.

Validation:

```bash
python3 examples/validation/validate_backend_blackoil_model.py
python3 examples/validation/validate_backend_spe1_tutorial.py
```

The validated SPE1 backend path currently matches the ADI tutorial for the first
three report steps within tight tolerances.

## Running SPE1 With The Torch Backend

Dense Torch backend:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --backend torch \
  --linear-solver dense \
  --steps 3
```

SciPy sparse linear solve after forming the dense Torch Jacobian:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --backend torch \
  --linear-solver scipy_sparse \
  --steps 1
```

CuPy sparse linear solve, if CuPy/CUDA are installed:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --backend torch \
  --linear-solver cupy_sparse \
  --steps 1
```

## Matrix-Free GMRES

A matrix-free Newton linear solve is available through Torch JVPs:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --backend torch \
  --linear-solver matrix_free_gmres \
  --gmres-scaling variable \
  --gmres-tolerance 1e-3 \
  --gmres-max-iterations 240 \
  --gmres-restart 120 \
  --steps 1
```

Validated behavior:

- Pure matrix-free GMRES with `--gmres-scaling variable` completes the first
  SPE1 timestep without dense fallback.
- It matches the dense backend first timestep to about `0.16 Pa` pressure and
  `0.85 Pa` BHP.
- It is still slower than dense for SPE1 because GMRES takes many JVP calls.

Fallback mode is also available:

```bash
python3 -m pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 \
  --backend torch \
  --linear-solver matrix_free_gmres_fallback \
  --steps 1
```

That mode attempts matrix-free GMRES and falls back to dense Newton when
unpreconditioned GMRES stalls. It is useful for diagnostics, not speed.

## Current Performance Reality

The fastest reliable paths today are still:

1. the standard ADI/SciPy PyMRST model
2. the dense Torch backend for small validation problems

The matrix-free path is important because it avoids materializing the full
Jacobian, which is required for larger GPU runs. But it needs better
preconditioning before it becomes faster than dense SPE1-sized solves.

## What Is Validated

Run these from the PyMRST root:

```bash
python3 examples/validation/validate_backend_tpfa.py
python3 examples/validation/validate_backend_oil_water.py
python3 examples/validation/validate_backend_blackoil_fluid.py
python3 examples/validation/validate_backend_blackoil_residual.py
python3 examples/validation/validate_backend_blackoil_model.py
python3 examples/validation/validate_backend_matrix_free.py
python3 examples/validation/validate_backend_spe1_tutorial.py
```

These cover:

- backend TPFA pressure solves
- oil-water Torch residual/Newton
- black-oil backend PVT tables
- live-oil SPE1 residual parity
- one-step backend black-oil Newton parity
- matrix-free GMRES helper correctness
- SPE1 backend tutorial parity for the first three report steps

## What Is Still Experimental

The following are not yet field-scale production paths:

- full 120-step SPE1 on pure matrix-free GMRES
- preconditioned matrix-free black-oil Newton
- sparse CUDA Jacobian assembly without dense intermediate formation
- full `GenericBlackOilModel` state-function graph on Torch tensors
- multisegment/VFP/facility-limit switching on the Torch backend at full MRST
  depth

The important milestone is that the backend now has real, validated physics
pieces rather than wrappers: backend PVT, TPFA, oil-water residuals, live-oil
black-oil residuals, SPE1 backend stepping, and matrix-free JVP-GMRES.
