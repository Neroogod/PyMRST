# PyMRST Layout Migration Plan

This document describes how to make PyMRST mirror the MRST folder/file layout
more closely without breaking Python packaging. The goal is not a cosmetic
shuffle. The goal is to make each ported Python function easy to trace back to
its MATLAB source file and easy to validate against MRST.

## Decision

Use a hybrid layout:

- Keep `pymrst/` as the Python package root.
- Keep underscores where MATLAB uses hyphens, because Python imports cannot use
  hyphenated package names.
- Mirror MRST's internal folder/file layout for functions that have been
  ported.
- Do not create empty files for unported MRST functions.
- Preserve the current public imports through `__init__.py` files and temporary
  compatibility modules where needed.

This means the target is:

```text
PyMRST/
  pymrst/
    gridprocessing/          # mirrors mrst-2024a-Geomech/gridprocessing
    params/                  # mirrors mrst-2024a-Geomech/params
    plotting/                # mirrors mrst plotting + mrst-gui where useful
    solvers/                 # mirrors mrst-2024a-Geomech/solvers
    utils/                   # mirrors mrst-2024a-Geomech/utils
    modules/                 # mirrors mrst-2024a-Geomech/modules
```

The unavoidable name translations are:

```text
ad-core              -> ad_core
ad-blackoil          -> ad_blackoil
ad-props             -> ad_props
ad-mechanics         -> ad_mechanics
ad-micp              -> ad_micp
co2lab-common        -> co2lab_common
co2lab-ve            -> co2lab_ve
co2lab-spillpoint    -> co2lab_spillpoint
co2lab-mit           -> co2lab_mit
dual-porosity        -> dual_porosity
mrst-gui             -> mrst_gui
```

## Rules For Every Move

1. Move structure only. Do not change numerical behavior in a layout PR.
2. Keep old import paths working until the whole project is migrated.
3. Add or update `__init__.py` exports immediately after every move.
4. Run the smallest relevant validation after each batch.
5. Keep each module directly runnable when that is already the PyMRST pattern.
6. Prefer one MRST function per Python file when the MATLAB source is one file.
7. Keep tightly coupled Python-only helpers together if splitting them makes the
   implementation harder to understand.
8. Update `GAPS.md`, `README.md`, and this file when a move changes the stated
   layout.

## Phase 0: Add A Source Map Before Moving Code

Start here. This creates the migration control surface.

Create `MRST_LAYOUT_MAP.md` or a machine-readable `layout_map.csv` with columns:

```text
mrst_path
pymrst_path
python_symbol
status
validation
notes
```

Example rows:

```text
gridprocessing/cartGrid.m
pymrst/gridprocessing/tensor_grid.py
cart_grid
ported
validate_phase1.py
kept with tensor_grid until Phase 4

params/rock/makeRock.m
pymrst/params/rock/make_rock.py
make_rock
ported
validate_phase2.py
move from params/rock.py in Phase 1
```

This map should become the answer to "where is the Python version of this MRST
file?"

Validation:

```bash
python3 examples/validation/validate_phase1.py
pytest -q
```

If the full suite is too slow, run the smallest phase validations affected by
the moved files.

## Phase 1: Split `params` First

Start with `params` because it is small, important, and currently differs from
MRST in the most obvious way. MRST has subfolders:

```text
params/
  rock/
  wells_and_bc/
```

Current PyMRST:

```text
pymrst/params/
  gravity_const.py
  rock.py
  wells_and_bc.py
```

Target PyMRST:

```text
pymrst/params/
  __init__.py
  gravity/
    __init__.py
    gravity.py
  rock/
    __init__.py
    make_rock.py
    perm_tensor.py
    pore_volume.py
  wells_and_bc/
    __init__.py
    add_bc.py
    add_source.py
    add_well.py
    compute_well_index.py
    pside.py
    fluxside.py
    vertical_well.py
```

Recommended first batch:

1. Create `pymrst/params/rock/`.
2. Move `Rock`, `make_rock`, `perm_tensor`, and `pore_volume` out of
   `pymrst/params/rock.py`.
3. Re-export them from `pymrst/params/rock/__init__.py`.
4. Keep `pymrst/params/__init__.py` exports unchanged.
5. Keep `pymrst/params/rock.py` temporarily as a compatibility wrapper if any
   imports still use `from pymrst.params.rock import ...`.

Recommended second batch:

1. Create `pymrst/params/wells_and_bc/`.
2. Move `BoundaryCondition`, `Source`, `Well`, `add_bc`, `add_source`,
   `add_well`, and `compute_well_index`.
3. Move `pside`, `fluxside`, and `vertical_well` if present in the current file.
4. Re-export from `pymrst/params/wells_and_bc/__init__.py`.
5. Keep `pymrst/params/wells_and_bc.py` temporarily as a wrapper.

Validation:

```bash
python3 pymrst/params/rock/make_rock.py
python3 pymrst/params/wells_and_bc/add_well.py
python3 examples/validation/validate_phase2.py
python3 examples/validation/validate_norne.py
```

Use `validate_norne.py` after the well move because Norne exercises reference
depth, connection pressure drop, gravity, wells, and corner-point geometry.

## Phase 2: Split `solvers`

MRST has individual solver files. PyMRST currently groups several functions.

Current PyMRST:

```text
pymrst/solvers/
  state.py
  trans.py
```

Target PyMRST:

```text
pymrst/solvers/
  __init__.py
  compute_trans.py
  get_face_transmissibility.py
  init_res_sol.py
  init_state.py
```

Order:

1. Split `compute_trans` into `compute_trans.py`.
2. Split `get_face_transmissibility` into `get_face_transmissibility.py`.
3. Split `init_state` and `init_res_sol`.
4. Re-export everything from `pymrst/solvers/__init__.py`.
5. Keep `trans.py` and `state.py` as temporary wrappers.

Validation:

```bash
python3 examples/validation/validate_phase2.py
python3 examples/validation/validate_phase3.py
python3 examples/validation/validate_phase5.py
```

## Phase 3: Split `utils` Conservatively

MRST `utils/` is broad. PyMRST should mirror only the numerical utilities it
has actually ported.

Current PyMRST:

```text
pymrst/utils/
  adi.py
  gridtools.py
  units.py
```

Target PyMRST:

```text
pymrst/utils/
  __init__.py
  ad_utils/
    __init__.py
    adi.py
  gridtools/
    __init__.py
    mcolon.py
    rldecode.py
    rlencode.py
  units/
    __init__.py
    units.py
```

Order:

1. Move `ADI`, `value`, and `init_variables_adi` under `utils/ad_utils/`.
2. Move `rldecode`, `rlencode`, and `mcolon` under `utils/gridtools/`.
3. Move unit constants under `utils/units/`.
4. Keep compatibility wrappers for `pymrst.utils.adi`, `pymrst.utils.gridtools`,
   and `pymrst.utils.units`.

Validation:

```bash
python3 pymrst/utils/ad_utils/adi.py
python3 pymrst/utils/gridtools/rldecode.py
python3 pymrst/utils/units/units.py
python3 examples/validation/validate_phase4.py
python3 examples/validation/validate_phase5.py
```

## Phase 4: Align `gridprocessing`

Do this after `params`, `solvers`, and `utils` because gridprocessing touches
many validation phases and imports.

Current PyMRST is already close at the folder level, but several files combine
multiple MRST functions or use Python names that do not show the MATLAB source
at a glance.

Target PyMRST:

```text
pymrst/gridprocessing/
  __init__.py
  cart_grid.py
  tensor_grid.py
  compute_geometry.py
  grid_structure.py
  extended_grid_structure.py
  process_grdecl.py
  cut_grdecl.py
  make_layered_grid.py
  pebi.py
  triangle_grid.py
  tetrahedral_grid.py
  private/
    __init__.py
    add_faces.py
    empty_grid.py
    remove_faces.py
  remove_cells.py
  extract_subgrid.py
```

Order:

1. Rename `geometry.py` to `compute_geometry.py`.
2. Split `tensor_grid.py` into `cart_grid.py` and `tensor_grid.py` if the code
   remains readable after the split. If not, keep a shared private helper.
3. Rename `grid.py` to `grid_structure.py`.
4. Move private MRST counterparts from `grid_editing.py` into
   `gridprocessing/private/`.
5. Move public grid editing functions such as `remove_cells` to public files.
6. Keep compatibility wrappers for old module names until all imports are
   updated.

Validation:

```bash
python3 examples/validation/validate_phase1.py
python3 examples/validation/validate_phase6.py
python3 examples/validation/validate_phase23.py
python3 examples/validation/validate_norne.py
```

Phase 1 protects Cartesian/tensor grids. Phase 6 protects corner-point grids.
Phase 23 protects unstructured grids and grid editing. Norne protects the real
corner-point field path.

## Phase 5: Split `deckformat`

Current PyMRST:

```text
pymrst/modules/deckformat/
  grdecl.py
  norne.py
  spe1.py
```

Target PyMRST:

```text
pymrst/modules/deckformat/
  __init__.py
  read_grdecl.py
  read_eclipse_deck.py
  spe1/
    __init__.py
    read_spe1_deck.py
  norne/
    __init__.py
    fetch_norne.py
    make_norne_grdecl.py
    norne_rock.py
    read_norne_grdecl.py
```

This one should follow the actual symbols in the existing files and the MATLAB
source, not a forced split. The main goal is to separate general GRDECL/deck
parsing from the Norne-specific dataset builder.

Validation:

```bash
python3 examples/validation/validate_phase5b.py
python3 examples/validation/validate_phase6.py
python3 examples/validation/validate_norne.py
```

## Phase 6: Split `co2lab` Into MRST Submodules

This is a larger move, but it will pay off because the current `co2lab` package
combines four MRST module families.

Current PyMRST:

```text
pymrst/modules/co2lab/
```

Target PyMRST:

```text
pymrst/modules/
  co2lab_common/
    __init__.py
    co2props.py
    h2props.py
    co2_brine.py
    co2_brine_props.py
  co2lab_ve/
    __init__.py
    ve.py
    ve_fluids.py
    ve_dissolution.py
  co2lab_spillpoint/
    __init__.py
    trapping.py
    node_trapping.py
  co2lab_mit/
    __init__.py
    atlas.py
    inventory.py
```

Keep `pymrst/modules/co2lab/` temporarily as a compatibility package that
re-exports the new modules. Remove it only after examples and validations no
longer import it directly.

Validation:

```bash
python3 examples/validation/validate_phase8.py
python3 examples/validation/validate_phase9.py
python3 examples/validation/validate_phase10.py
python3 examples/validation/validate_phase11.py
python3 examples/validation/validate_phase12.py
python3 examples/validation/validate_phase23.py
```

Also run any CO2 workflow examples that import `pymrst.modules.co2lab`.

## Phase 7: Align `ad_core`

Do not start here. `ad_core` is foundational and many modules import it.

Current PyMRST:

```text
pymrst/modules/ad_core/
  model.py
  nonlinear_solver.py
  operators.py
  properties.py
  schedule.py
  state_functions.py
```

Target PyMRST:

```text
pymrst/modules/ad_core/
  __init__.py
  models/
    __init__.py
    physical_model.py
    reservoir_model.py
    facility_model.py
  solvers/
    __init__.py
    nonlinear_solver.py
  statefunctions/
    __init__.py
    state_function.py
    state_function_grouping.py
    pvt_property_functions.py
    flow_property_functions.py
    flow_discretization.py
  utils/
    __init__.py
  schedule/
    __init__.py
    schedule.py
  operators/
    __init__.py
    tpfa_operators.py
```

Order:

1. Move state-function graph classes first.
2. Move property groupings second.
3. Move model/facility classes third.
4. Move solver and schedule last.
5. Keep wrappers for `pymrst.modules.ad_core.model`,
   `pymrst.modules.ad_core.properties`, and
   `pymrst.modules.ad_core.state_functions`.

Validation:

```bash
python3 examples/validation/validate_phase24.py
python3 examples/validation/validate_phase5.py
python3 examples/validation/validate_phase5b.py
python3 examples/validation/validate_phase27.py
python3 examples/validation/validate_norne.py
```

## Phase 8: Align `ad_blackoil`, `ad_props`, And `incomp`

These modules sit directly on top of `ad_core`, so move them after Phase 7.

Targets:

```text
pymrst/modules/ad_props/
  fluids/
    init_simple_adi_fluid.py
    blackoil_fluid.py
  props/
    tables.py

pymrst/modules/ad_blackoil/
  models/
    oil_water_model.py
    blackoil_model.py
    generic_blackoil_model.py

pymrst/modules/incomp/
  fluids/
    init_simple_fluid.py
  solvers/
    incomp_tpfa.py
  transport/
    explicit_transport.py
    implicit_transport.py
```

Validation:

```bash
python3 examples/validation/validate_phase3.py
python3 examples/validation/validate_phase5.py
python3 examples/validation/validate_phase5b.py
python3 examples/validation/validate_phase24.py
```

## Phase 9: Align Geomechanics Modules

Move these only after the flow stack is stable.

Targets:

```text
pymrst/modules/vemmech/
  vem/
    vem.py
    biot.py
  operators/
    strain.py
    stress.py

pymrst/modules/ad_mechanics/
  models/
    mechanics.py
    poromech.py
    mech_blackoil.py
  solvers/
    linear_solver.py
  adjoint/
    adjoint.py
  examples/
    mandel.py
```

Validation:

```bash
python3 examples/validation/validate_phase7.py
python3 examples/validation/validate_phase27.py
python3 examples/validation/validate_norne.py
python3 examples/run_norne_geomech.py
```

Use the mini Norne run first. Run full Norne only when the structural work is
otherwise clean.

Progress note: `ad_mechanics` now has MRST-shaped compatibility packages for
`models/`, `models/fully_coupled/`, `models/splitting/`, `utils/`, `solvers/`,
`adjoint/`, and `examples/`. The adjoint implementation file was renamed from
`adjoint.py` to `adjoint_impl.py` so the new `adjoint/` package can mirror the
MRST layout without shadowing the implementation. The public imports still
expose the same symbols.

## Phase 10: Align Remaining Small Modules

After the heavy core is stable, migrate the smaller modules one by one.

Recommended order:

1. `diagnostics`
2. `upscaling`
3. `streamlines`
4. `optimization`
5. `sequential`
6. `dual_porosity`
7. `geothermal`
8. `ensemble`
9. `compositional`
10. `ad_micp`
11. `hfm`

Progress note: `compositional` now has MRST-shaped compatibility packages for
`fluids/`, `models/eos/`, `models/overall/`, `models/overall/sequential/`, and
`utils/`. The validated implementations remain in the original flat files for
now; the new files are landing modules that preserve old imports while making
MRST source paths discoverable.

Progress note: `sequential` now has MRST-shaped compatibility packages for
`models/` and `utils/`, delegating to the existing `seq_model.py`
implementation.

Progress note: `diagnostics` now has an MRST-shaped `utils/` package for the
ported TOF/tracer, F-Phi, Lorenz, sweep, and well-pair landing files. The
validated implementation remains in `flow_diagnostics.py` for now.

Progress note: `upscaling` now has MRST-shaped compatibility packages for
`coarsegrid/` and `flow/`, covering the validated Cartesian partition,
coarse-grid generation, fixed TPFA permeability upscaling, volume-weighted
porosity, and rock upscaling slice. Periodic, relperm, and capillary
upscaling remain unported.

Progress note: `streamlines` now has an MRST-shaped `utils/` package for
`findStreamlineNeighborshipCart` and `getInteriorFluxesStreamlines`. The
validated Pollock tracer remains in `pollock.py`, which already mirrors the
top-level MRST `pollock.m` entry point.

Progress note: `optimization` now has MRST-shaped compatibility packages for
the ported adjoint/NPV slice: `objectives/NPVOW`, `utils/computeGradientAdjointAD`,
`utils/evalObjective`, `utils/simulationSolverFun`, and a PyMRST
projected-gradient `optim/optimize_rates` landing file. This is not the full
MRST optimizer framework; the module remains finite-difference verified
against PyMRST's validated oil-water AD model.

Progress note: `dual_porosity` now has MRST-shaped compatibility packages for
`ad_models/`, `ad_models/equations/`, `transfer_models/`, and
`transfer_models/shape_factor_models/`. The validated implementation remains
split across `dp_model.py`, `transfer.py`, and `shape_factors.py`.

Progress note: `geothermal` now has MRST-shaped compatibility packages for
`eos/water_density/`, `eos/water_viscosity/`, `fluids/`, `models/`, and the
implemented `statefunctions/` slices. The validated EOS remains in
`water_props.py`; the coupled mass-energy implementation remains in
`geothermal_model.py`. Wellbore, facility/group-control, and the full MRST
statefunction graph remain unported.

Progress note: `ensemble` now has MRST-shaped compatibility packages for
`ensemble/`, `ensemble/samples/`, `ensemble/utils/`, `history_matching/`, and
`history_matching/utils/`. Because those names conflict with Python's old flat
`ensemble.py` and `history_matching.py`, the implementation files now live in
`ensemble_impl.py` and `history_matching_impl.py`; the public package imports
still expose the same symbols.

Progress note: `ad_micp` now has MRST-shaped compatibility packages for
`models/`, `fluids/`, and `utility/`, covering `MICPModel`, `equationsMICP`,
clogging checks, and the implemented flux/dispersion/coupling pieces.
`MICPMechModel` remains out of scope because this MRST tree does not contain
the corresponding `equationsMICPMech.m`.

Progress note: `hfm` now has MRST-shaped compatibility packages for the
ported 2D EDFM slice: `gridding/`, `processing/`, `utils/`, and `operators/`.
The implementation remains in `edfm2d.py`. 3D EDFM, projection-based
`hfm_v2`, fracture mechanics/TEDFM, multiscale/Rsb support, and plotting
remain unported.

Phase 10 status: the small-module layout pass is complete for all ported
symbols in this batch. Remaining gaps are implementation gaps, not just layout
gaps.

These modules have fewer imports into the rest of the package, so they are
lower risk once core imports are stable.

Validation:

```bash
python3 examples/validation/validate_phase11.py
python3 examples/validation/validate_phase12.py
python3 examples/validation/validate_phase13.py
python3 examples/validation/validate_phase14.py
python3 -m pymrst.modules.optimization.adjoint
python3 examples/validation/validate_phase15.py
python3 examples/validation/validate_phase16.py
python3 -m pymrst.modules.ensemble.ensemble_impl
python3 -m pymrst.modules.ensemble.history_matching_impl
python3 -m pymrst.modules.ad_micp.micp_fluid
python3 -m pymrst.modules.ad_micp.micp_model
python3 examples/validation/validate_phase7b.py
python3 -m pymrst.modules.hfm.edfm2d
python3 examples/validation/validate_phase7c.py
python3 examples/validation/validate_phase17.py
python3 examples/validation/validate_phase18.py
python3 examples/validation/validate_phase20.py
python3 examples/validation/validate_phase21.py
python3 examples/validation/validate_phase22.py
python3 examples/validation/validate_phase7b.py
python3 examples/validation/validate_phase7c.py
```

## Phase 11: Align `plotting`

Plotting has a looser MRST mapping because PyMRST uses PyVista and MRST uses
MATLAB graphics. Mirror the MRST names only where the conceptual equivalent is
clear.

Target:

```text
pymrst/plotting/
  __init__.py
  plot_grid.py
  plot_cell_data.py
  plot_well.py
  extensions/
    __init__.py
    browser.py
    plot_inventory.py
    plot_saved_states.py
    plot_streamlines.py
    plot_toolbar.py
    plot_ve.py
```

Validation:

```bash
python3 examples/validation/validate_phase12.py
python3 examples/validation/validate_phase19.py
```

If GUI dependencies are not installed, run non-GUI import tests and keep the GUI
validation optional.

## Phase 12: Remove Compatibility Wrappers

Only remove old wrapper modules after:

1. All PyMRST imports use the new layout.
2. All examples use the new layout.
3. All validation scripts pass.
4. `README.md`, `GAPS.md`, and this file describe the new layout.

Search commands:

```bash
rg "pymrst\.params\.rock"
rg "pymrst\.params\.wells_and_bc"
rg "pymrst\.gridprocessing\.geometry"
rg "pymrst\.modules\.co2lab"
rg "pymrst\.modules\.ad_core\.model"
```

Remove one wrapper at a time and run the relevant validations.

## Recommended Starting Point

Start with Phase 0 and Phase 1 only:

1. Add the layout map.
2. Split `params/rock.py`.
3. Preserve old imports.
4. Run Phase 2 validation.
5. Split `params/wells_and_bc.py`.
6. Run Phase 2 and Norne validation.

This gives a useful proof of the migration pattern without touching the AD
stack or CO2 stack. If that goes cleanly, continue to `solvers`, `utils`, then
`gridprocessing`.

## Migration Progress

Completed layout-only migrations:

| Area | Status | Compatibility kept | Validation |
|---|---|---|---|
| `params/rock.py` -> `params/rock/` | done | `from pymrst.params.rock import ...` | module self-tests, `validate_phase2.py`, `validate_phase22.py` |
| `params/wells_and_bc.py` -> `params/wells_and_bc/` | done | `from pymrst.params.wells_and_bc import ...` | module self-tests, `validate_phase2.py`, `validate_phase22.py` |
| `solvers/trans.py` implementation -> `compute_trans.py`, `get_face_transmissibility.py` | done | `pymrst.solvers.trans` wrapper | module self-tests, `validate_phase2.py`, `validate_phase3.py`, `validate_phase22.py` |
| `solvers/state.py` implementation -> `init_res_sol.py`, `init_state.py`, `state_types.py` | done | `pymrst.solvers.state` wrapper | module self-tests, `validate_phase2.py`, `validate_phase3.py` |
| `utils/gridtools.py` -> `utils/gridtools/` | done | `from pymrst.utils.gridtools import ...` | module self-tests, `validate_phase1.py`, `validate_phase2.py`, `validate_phase22.py` |
| `utils/adi.py` -> `utils/adi/` | done | `from pymrst.utils.adi import ...` | package/module self-tests, `validate_phase4.py`, `validate_phase5.py`, `validate_phase5b.py`, `validate_phase24.py`, `validate_phase27.py` |
| `utils/units.py` -> `utils/units/` | done | `from pymrst.utils.units import ...` | package/module self-tests, `validate_phase4.py`, `validate_phase5.py`, `validate_phase5b.py` |
| `gridprocessing/grid.py` implementation -> `grid_structure.py` | done | `pymrst.gridprocessing.grid` wrapper | module self-tests, `validate_phase1.py`, `validate_phase6.py`, `validate_phase23.py`, `validate_norne.py` |
| `gridprocessing/geometry.py` implementation -> `compute_geometry.py` | done | `pymrst.gridprocessing.geometry` wrapper | module self-tests, `validate_phase1.py`, `validate_phase6.py`, `validate_phase23.py`, `validate_norne.py` |
| `gridprocessing/grid_editing.py` implementation -> `private/` + `remove_cells.py` | done | `pymrst.gridprocessing.grid_editing` wrapper | module self-tests, `validate_phase23.py` |
| `gridprocessing/cartGrid.m` landing file -> `cart_grid.py` | done | `tensor_grid.cart_grid` still available | module self-tests, `validate_phase1.py` |
| `modules/deckformat/*.py` -> `deckformat/{grdecl,spe1,norne}/` | done | `pymrst.modules.deckformat.grdecl`, `.spe1`, `.norne` package exports | module self-tests, `validate_phase5b.py`, `validate_phase6.py`, `validate_phase24.py`, `validate_norne.py` |
| `modules/co2lab/*.py` -> `co2lab_common/`, `co2lab_ve/`, `co2lab_spillpoint/`, `co2lab_mit/` | done | `pymrst.modules.co2lab` remains the implementation and old import surface | compileall, import validation, CO2 phase validations; atlas data validation skipped when `MRST_DATA_DIR` is unset |
| `modules/ad_core/*.py` -> `ad_core/{models,simulators,statefunctions,utils,props}/` | done | `pymrst.modules.ad_core.model`, `.nonlinear_solver`, `.state_functions`, `.properties`, `.operators`, `.schedule` remain implementation surfaces | compileall, import validation, wrapper smoke tests, AD core self-tests, `validate_phase5.py`, `validate_phase5b.py`, `validate_phase7.py`, `validate_phase7b.py`, `validate_phase7c.py`, `validate_phase24.py`, `validate_phase27.py`, `validate_norne.py` |

Next recommended migration:

1. Start `ad_blackoil`, splitting public entry points around the MRST
   black-oil model families while keeping current imports working.
2. Map `generic_model.py`, `oil_water_model.py`, and `blackoil_model.py`
   before moving any implementations.
3. Validate with `validate_phase5.py`, `validate_phase5b.py`,
   `validate_phase24.py`, and `validate_norne.py`.

## Definition Of Done

The layout migration is done when:

- Every ported MRST function has a documented PyMRST path.
- The PyMRST path mirrors the MRST path where Python allows it.
- Old compatibility wrappers have either been removed or explicitly documented.
- `README.md` no longer claims a stricter mirror than the code actually has.
- `GAPS.md` records only meaningful implementation gaps, not layout confusion.
- The validation suite passes against the MATLAB reference dumps.
