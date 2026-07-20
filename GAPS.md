# What PyMRST does not have (yet)

A direct comparison of this package against the reference MATLAB tree in
`../mrst-2024a-Geomech`, done by reading both trees rather than the roadmap.
The roadmap in `README.md` tracks what has been *built*; this file tracks what
has *not*, so the two together say where the port actually stands.
`ROADMAP.md` says how what is missing gets built - in what order, and how each
piece is validated.

The one-line summary, as of Phase 336: **the port is no longer a uniform
slice - it is uneven.** Compositional, black-oil, poromechanics and
dual-porosity now carry broad API surfaces; about half the tree still carries
the single code path needed to reproduce one MATLAB-validated case. And the
capability gaps are almost exactly where they were eighty phases ago: TPFA is
still the only discretization, and there is still no multiscale solver, no
mesh generator, and no EOR.

An earlier version of this file summarised the port as "a deep vertical slice,
not a wide port" and backed it with a table that has since gone badly stale -
it credited `ad-core` with 6 Python files against 345 MATLAB ones, and
`compositional` with 5 against 96. Both were off by an order of magnitude, and
the `co2lab` MATLAB count was too low by 81 files. The table below is
regenerated from the trees.

## Where the port is broad, and where it is thin

`.m` files in MATLAB vs `.py` files here, both counted from the trees. File
counts are a weak proxy in *both* directions - MRST puts one function per
file, which inflates its side, while PyMRST groups related functions into
larger modules, which deflates its own - so the line count is given as a
second signal.

| Module | MRST `.m` | PyMRST `.py` | PyMRST lines |
|--------|-----------|--------------|--------------|
| `compositional` | 96 | **129** | 22,857 |
| `ad-blackoil` | 57 | **60** | 5,631 |
| `ad-mechanics` | 44 | **59** | 4,683 |
| `dual-porosity` | 19 | **20** | 1,392 |
| `vemmech` | 72 | 55 | 11,786 |
| `ad-core` | 345 | 60 | 12,426 |
| `co2lab-*` (4 dirs) | 216 | 27 | 5,209 |
| `geothermal` | 105 | 32 | 1,320 |
| `hfm` | 180 | 19 | 1,437 |
| `ensemble` | 68 | 19 | 843 |
| `deckformat` | 112 | 15 | 956 |
| `incomp` | 52 | 14 | 1,648 |
| `ad-micp` | 77 | 13 | 702 |
| `diagnostics` | 143 | 9 | 683 |
| `mpsaw` | 91 | 7 | 439 |
| `hfm_v2` | 180 | 4 | 60 |
| `dfm` | 63 | 1 | 32 |
| `shale` | 48 | 1 | 32 |

Read it in three bands:

- **At or past file parity** (`compositional`, `ad-blackoil`, `ad-mechanics`,
  `dual-porosity`) - these are wide ports now. "One code path per module" has
  not described them for a long time.
- **Deep but not wide** (`ad-core`, `vemmech`, `co2lab`) - heavy in lines,
  light in files. Real breadth, organised into fewer and larger modules than
  MRST's one-function-per-file layout, so the file ratio understates them.
  `co2lab` remains ported end to end across all four MATLAB dirs (`-common`,
  `-ve`, `-spillpoint`, `-mit`).
- **Still a slice** (`diagnostics`, `deckformat`, `geothermal`, `hfm`,
  `ad-micp`, `ensemble`, `incomp`) - the port reaches in far enough to match
  one reference and stops. This was once the description of the whole tree; it
  now describes about half of it.

`hfm_v2`, `dfm` and `shale` sit in a fourth band of their own: they are
**re-export shims**, 25-32 lines each, forwarding to `hfm`'s validated EDFM
kernels under MRST's other module names. They add no independent capability
and should not be read as three more ported modules. `mpsaw` is similar in
spirit - 439 lines, of which the `utils`, `models` and `statefunctions`
subpackages are 5-8 line stubs.

## What the last eighty phases bought

Phases 261-336 - the most recent block, and roughly a quarter of the port's
total phase count - are almost entirely **API-surface parity**, not new
physics. Reading their validators end to end: utility entry-point aliases,
`Contents.m` catalog parity, MRST-layout compatibility modules, public-surface
naming, property aliases, constructor overloads, and getter/setter parity on
`FacilityModel` and `EquationOfStateModel`.

This is real work, and it is why `compositional` now sits past file parity.
It is worth being precise about what it does and does not close:

- It **does** close the gap between "PyMRST can compute this" and "MRST code
  reads across unchanged" - the thing that makes the MATLAB examples and the
  two textbooks map onto this tree.
- It **does not** close any capability gap in this file. Every item under
  "Whole modules not ported" was missing at Phase 260 and is missing now, and
  every function under "Core gaps" below was re-checked at Phase 336 and is
  still absent.

Both halves matter. A port that computes the right answer through an alien API
is hard to use; a port with a familiar API and no new physics is still missing
the physics.

## Core gaps (confirmed by grep - zero Python counterparts)

These matter most because the roadmap treats the core as finished.

### Grid construction
Have (since Phase 23): `triangle_grid`, `tetrahedral_grid`, `pebi`,
`make_layered_grid` - simplex grids, Voronoi/PEBI duals, and layered
extrusion, all validated bit-exact against MATLAB. Still missing:

- `hexahedralGrid`, `tessellationGrid`, `glue2DGrid`, `get2DGridFromFaces`
- the meshing pipeline that generates point sets (`upr`, `gmsh`, `triangle`
  modules) - `pebi` consumes a triangulation; nothing here produces one
  beyond scipy's Delaunay

### Grid editing
Have: `extract_subgrid`, and (Phase 23) `add_faces`, `remove_faces`,
`remove_cells`, `empty_grid`. Missing:

- `makeInternalBoundary` / `removeInternalBoundary`
- `sliceGrid`, `refineGrdecl`, `removePinch` / `processPINCH`
- `processNNC`, `splitDisconnectedGrid`, `repairNormals`, `removeShortEdges`,
  `findMatchingFaces`

### Rock / petrophysics (`params/rock`)
Have: `make_rock`, `perm_tensor`, `pore_volume`, `gaussian_field`,
`log_norm_layers`. Missing:

- `grdecl2Rock`, `permeabilityConverter`

### Wells and boundary conditions (`params/wells_and_bc`)
Have: `add_well`, `add_bc`, `add_source`, `compute_well_index`, `cell_dims`,
`vertical_well`, `pside`, `psideh`, `fluxside`, and (since the Norne cross-check) the
**connection pressure drop** `rho_mix g dZ` between a well's bhp datum and its
perforations. Missing:

- ~~`psideh` - the hydrostatic variant of `pside`~~ **Done** (Phase 225):
  selected side faces receive `ref_pressure + omega g\cdot(x-x_ref)`, where
  `omega` is MRST's mobility-weighted density `sum((kr/mu) rho)/sum(kr/mu)`.
  The port supports uniform saturations, side aliases, the existing PyMRST
  in-plane logical ranges, and the MRST perpendicular `range` option as
  `search_range`/`range`. **Phase 226** also exposes this through compositional
  forcing controls as `psideh`/`pSideH`/`HydrostaticSide`. **Phase 227** adds a
  direct MATLAB-backed `psideh` regression, comparing selected faces,
  centroids, saturations, types, mobility-weighted density, and hydrostatic
  pressure values against MRST.
- ~~the wellbore *topology* sum in `updateConnectionPressureDrop`: PyMRST's cdp
  is a single hydrostatic column per perforation, which is exact for
  one-perforation wells and an approximation for a well with many.~~ **Done**
  (Phase 49): the column is now accumulated segment by segment, each segment
  weighing the cumulative stream that flows past it (MRST's
  `updateConnectionPressureDropState`, which is the one `GenericFacilityModel`
  actually calls). Validated on two five-perforation vertical wells - the
  wellbore column itself, not just its consequences - to machine precision.
  This was the port's only *known-wrong* path: no MATLAB comparison here had
  ever used a well with more than one perforation.
- ~~VFP tables and multisegment-well equations~~ - partially integrated:
  `VFPTable` evaluates MRST-style producer/injector BHP tables and
  `FacilityModel` can close THP/VFP-controlled wells explicitly from current
  surface rates. `MultisegmentWell` stores MRST-style segment topology, and
  `assemble_multisegment_equations` ports the core node mass equations,
  segment pressure equations, component-fraction closure, and MRST's
  `wellBoreFriction` empirical pressure-loss law. `GenericBlackOilModel`
  automatically packs the extra MS well primary variables (`pN`, node
  fractions, `vmix`) and appends the matching equations; Phase 109 validates
  the schedule-level packed variable counts/topology against MRST's converted
  MS-well layout and checks the resulting PyMRST Newton system is square.

### Solvers
Have: `compute_trans`, `get_face_transmissibility`, `init_state`,
`init_res_sol`. Missing:

- `accumulateCartesianMultipliers` (transmissibility multipliers)
- `initWellSol` (folded into `init_state`, not standalone)

### Utils
Have: `ADI`, `value`, `init_variables_adi`, `rldecode`/`rlencode`/`mcolon`,
units. Missing, numerically relevant:

- flux/velocity reconstruction: `faceFlux2cellVelocity`, `cellFlux2faceFlux`

(`interpTable` / `dinterpTable` *are* present, as `interp_table` / `_interp1`
in `modules/ad_props/tables.py`, AD-transparent and shared by `co2lab`.)

The rest of `utils/` is MATLAB session plumbing (`mrstModule`, `merge_options`,
`mrstPath`, ...) with no Python equivalent needed.

## Depth gaps inside "done" modules

- **`vemmech`** - linear elasticity + linear Biot, plus (Phase 27) the strain
  and stress operators and `invCi`. The VEM operator path now supports both
  3D and 2D plane-strain grids (`calculateQF`, `VEM2D_div`, 2D stiffness,
  strain/stress operators). Standalone stress utilities
  (`calculateStressVEM`, `calStressEigsVEM`) and batched eigensolvers
  (`multiEig`, `multiSymmEig`) are matched by `validate_phase68.py`. The core
  TEDFM VEM wrappers (`calculateQFTEDFM`, `calculateQCTEDFM`, `C2DTEDFM`,
  `VEM3D_divTEDFM`, `VEM_divTEDFM`, `VEM_assembleTEDFM`) are matched by
  `validate_phase69.py` for `G.Matrix` wrapper grids; the
  `VEM_linElastTEDFM` solve wrapper is matched by `validate_phase70.py`.
  The `grid_utils` helpers (`flipGrid`, `padGrdecl`, `verticalGrdecl`) are
  matched by `validate_phase71.py`; PyMRST handles x-only/y-only padding even
  though MRST's `padGrdecl` has an uninitialized `grdecl_new` defect on those
  direct paths. The deformed plotting helpers are ported as backend-agnostic
  geometry builders (`plotGridDeformed`, `plotCellDataDeformed`,
  `plotFaceDataDeformed`, `plotNodeDataDeformed`, `plotFaces2D`) and matched
  by `validate_phase72.py`. The first AD VEM substrate slice is ported as an
  eager numeric plus first-order ADI coefficient tensor layer
  (`SparseMultiArray`, `kron_ADI`, `face_node_tensor`, `Enu2C_AD`) and matched
  by `validate_phase73.py`/`validate_phase94.py`. Phase 95 adds a pure-Python
  lazy expression graph (`LazySparseTensor`, `tbinaryop`, `tcontract`) and an
  exact second-order AD substrate (`ADI2`, `hessian_ad2`) checked by
  `validate_phase95.py`. MRST's own lazy tensor contraction/binary ops are
  MEX-backed and cannot compile locally on this MATLAB/Xcode target
  (`apple-m4` target error), so compiled-backend identity/performance parity
  remains pending. The
  numeric `VEM_assemble_AD`/`VEM_linElast_AD` path is now present and matched
  by `validate_phase78.py` for numeric E/nu, AD-shaped `SparseMultiArray`
  BC/load inputs, reduced solve, `no_solve`, and `disc` operators. Material
  AD derivatives are present as `compute_ax_derivs`/`Ax_derivs` for 3D
  (`validate_phase79.py`) and 2D (`validate_phase81.py`) inputs, checked
  against central finite differences. Full compiled-MEX sparse-tensor AD
  factor-graph parity remains pending, but the calibration
  RHS/BC/material/load/objective derivative path is now covered by explicit
  analytic/eager derivative blocks.
  `setup_objective_fun` now exposes an eliminated
  objective (`validate_phase80.py`) by solving mechanics and using the MRST
  adjoint structure. The reduced residual helper behind this path is checked
  by `validate_phase82.py`. Boundary-condition derivative columns are checked
  by `validate_phase85.py`. Material E/nu closures now preserve AD
  derivatives (`validate_phase83.py`), and stiffness-control columns in the
  calibration adjoint use `Ax_derivs` (`validate_phase84.py`). Load/density
  residual columns are checked by `validate_phase86.py`. The objective
  displacement partial `oval_dd` now uses the VEM stress operator rather than
  finite-differencing every mechanical degree of freedom (`validate_phase87.py`).
  The control-side objective partial `oval_du` is split into analytic prior,
  analytic boundary-control stress contributions, and localized stress
  sensitivities (`validate_phase88.py`). The local stress-observation
  derivative bridge is now analytic for `Smax`, `Smin`, `Sv`, and orientation
  mismatch (`validate_phase89.py`). `objfun` keeps MRST-compatible empty
  Hessian outputs, while PyMRST exposes an optional ADI2-propagated
  eliminated-objective Hessian through `objfun_with_hessian`
  for constant-material boundary-control calibration cases
  (`validate_phase90.py`). `calibrate_model`, `optimize_mech`, and
  `produce_syndata` now provide a runnable calibration workflow with
  MRST-style result fields (`validate_phase91.py`). The default workflow uses
  an SR1 trust-region optimizer slice patterned after MRST `optimizeSR1`,
  checked on a quadratic trust-region case (`validate_phase92.py`). Backup-file
  continuation and progress callbacks are checked by `validate_phase93.py`.
  The
  first numeric calibration helper slice (`Sigma2SmaxSminTheta`,
  `identify_boundary_elements`, `set_material_fun`) is matched by
  `validate_phase74.py`, and the calibration boundary-condition helper
  (`set_bcfun`) is matched by `validate_phase75.py` using a dense
  source-equivalent reference because MRST's callable triggers the local MEX
  tensor-contraction blocker. The numeric value part of calibration
  `calculate_stress` is matched by `validate_phase76.py`; full AD Jacobians for
  the calibration objective are still pending. The numeric Gaussian objective
  value slice of `setup_objective_fun` is matched by `validate_phase77.py`,
  its eliminated mechanics objective is checked by `validate_phase80.py`; the
  stress-operator displacement objective partial is checked by
  `validate_phase87.py`, and the structured control objective partial is
  checked by `validate_phase88.py`; local stress-observation derivatives are
  checked by `validate_phase89.py`; optional ADI2 eliminated Hessian output is
  checked by `validate_phase90.py`; material-control Hessians for K/G/rho
  controls are checked by `validate_phase96.py`; the calibration workflow driver is checked by
  `validate_phase91.py`; the SR1 trust-region optimizer slice is checked by
  `validate_phase92.py`; backup/resume and progress hooks are checked by
  `validate_phase93.py`; first-order ADI tensor coefficient propagation is
  checked by `validate_phase94.py`; pure-Python lazy tensor graph evaluation
  and exact second-order AD Hessians for scalar expressions are checked by
  `validate_phase95.py`. MATLAB-style deformed plotting handles are checked by
  `validate_phase97.py`, and broader triangle/PEBI/tetrahedral VEM mechanics
  invariants are checked by `validate_phase98.py`. Phase 102 adds a global
  elastoplastic VEM Newton driver for 2D small-strain von-Mises mechanics,
  including elastic-limit and plastic load-ramp validation; Phase 104 extends
  the same VEM plastic stress-correction path to 3D Cartesian/tetrahedral
  mechanics and checks the elastic-limit and yielded-load branches.
  water, oil-water, and black-oil fully coupled/fixed-stress branches of
  MRST's `run2DCase.m` are matched by `validate_phase63.py`,
  `validate_phase64.py`, `validate_phase62.py`, `validate_phase65.py`,
  `validate_phase66.py`, and `validate_phase67.py`.
  Still missing: compiled-backend identity/performance parity with MRST's
  MEX-backed sparse-tensor engine and bit-for-bit identity with MRST's full
  local MPSA-W Schur/tensor assembly.
- **`hfm`** - 2D EDFM flow is ported and the first EDFM/TEDFM mechanics
  workflow coupling is now present (`validate_phase99.py`): matrix mechanics is
  solved through the TEDFM VEM wrapper and embedded fracture cells receive
  normal-strain/aperture-change fields. Phase 103 adds a focused 3D planar
  EDFM/projection workflow and mechanics-driven 2D fracture-tip propagation
  update. Phase 106 adds MRST-shaped `hfm_v2`, `dfm`, and `shale`
  compatibility landing modules plus toughness-driven propagation helpers.
  Phase 107 compares the 3D planar EDFM projection against a MATLAB MRST HFM
  dump for physical fracture volume/area and matrix-NNC consistency. Still
  missing: exact MRST fracture triangulation/workflow identity for the larger
  `hfm_v2`, DFM, hwu-fractures/shale examples, multiscale/Rsb support, and
  fracture-propagation physics beyond the local aperture/toughness criteria.
- **`compositional`** - Peng-Robinson EOS + two-phase flash (Phase 13), the
  **phase density and viscosity** (Phase 50), and now (Phase 51) the
  **flash-to-flow-props** step that turns a flash result into exactly what a
  per-component conservation equation reads:
  - Phase 50 - `compositional/properties.py` ports MRST's
    `CompositionalPropertyModel`: `rho = M(x)/(ZRT/p)` with the optional volume
    shift, and the Lohrenz-Bray-Clark viscosity (in its native Rankine/psia/cP
    units). `validate_phase50.py` matches MATLAB at machine precision (density
    to 1.7e-16, viscosity to 2.7e-15) across 84 liquid/vapour/two-phase
    conditions, with and without a volume shift.
  - Phase 51 - `compositional/flow_props.py` ports `computeTwoPhaseFlowProps`:
    `getMassFraction` (moles -> mass, because the accumulation term is written
    in mass), `computeSaturations` (the *molar* liquid fraction `L` -> a
    *volume* saturation, which with no shift is `L Z_L / (L Z_L + (1-L) Z_V)`),
    and the `(p,T,z) -> {xM, yM, sL, sV, rhoL, rhoV, muL, muV}` pipeline they
    feed. `validate_phase51.py` replays these formulas on MRST's own flash and
    matches MATLAB to machine precision (mass fractions exact, saturations to
    8.7e-19), including the volume-shift saturation split.
  - Phase 52 - `compositional/eos_ad.py` ports the *differentiated* flash
    (`getPhaseFractionAsADI` / `getPhaseFractionDerivativesPTZ`). A
    fully-implicit model differentiates against pressure and `z`, but the flash
    is an *implicit* function of `(p,T,z)`, so `dL, dx, dy` come from the
    implicit function theorem `ds/du = -(dF/ds)^{-1}(dF/du)` on the equilibrium
    system `F` (component balances, equal fugacity, phase closure). The two
    Jacobians are obtained by evaluating `F` twice under PyMRST's forward-mode
    AD (seeding `u`, then `s`); the system is block-diagonal by cell, so the
    solve is a batch of small dense `(2n+1)x(2n+1)` systems. The cubic root's
    derivative is attached analytically (`setZDerivatives`,
    `dZ = -(dE2 Z^2 + dE1 Z + dE0)/(3Z^2 + 2E2 Z + E1)`). `flash_ad` composes
    these with the primaries' Jacobians (`addDerivativePTZ`, including the
    `z_n = 1 - sum z_i` constraint). `validate_phase52.py` matches MRST's own
    `getPhaseFractionDerivativesPTZ` at machine precision (dL, dx, dy all to
    ~1e-15 relative across 35 two-phase cells); the self-tests cross-check every
    derivative against finite differences of the flash.

  - Phase 53 - `compositional/model.py` ports the overall-composition **flow
    model** (`equationsCompositional`, no-water branch): per-component mass
    conservation with phase-potential upwinding, primaries `(p, z_1..z_{n-1})`,
    and a fully-implicit Newton `step()`. Every underlying piece is wired in:
    `flash_ad` + `phase_compressibility_ad` for the ADI branch of
    `computeTwoPhaseFlowProps`, `flow_props` for mass fractions and saturations,
    `properties` for density and viscosity. `validate_phase53.py` runs the same
    closed 1D equilibration case as MRST's `OverallCompositionCompositionalModel`
    (10 cells, 5×20 days, pressure gradient, no wells): pressure to **3.7e-4
    bar**, composition to **7.6e-7**, saturation to **3.7e-6** (a small
    systematic offset from step 1, not drift; component mass conserved to
    **1.5e-11**). Recomputing `sO` from MRST's own `p,z` matches MRST's stored
    saturation to **5e-10**, confirming the saturation formula. Module
    self-tests include a finite-difference Jacobian check, closed-domain mass
    balance, and single-phase stability.

  - Phase 54 - **component-aware wells** in `compositional/model.py`: the
    standard MRST well model (surface phase rates `qOs`, `qGs`, `bhp` as
    primaries) plus `getExtraWellContributions` per-component mass sources from
    phase surface rates (`cq_s * rhoS`) and injection mass fractions
    (`W.components` via `getMassFraction`). `Well.components` added to
    `params/wells_and_bc/types.py`. `validate_phase54.py` runs a 1D
    methane-rich gas injector / BHP producer (10 cells, 5×30 days): pressure to
    **7.9e-6 bar**, composition to **6.9e-7**, saturation to **1.0e-6**, well
    rates to **2e-9 m³/s**, BHP to **7.9e-6 bar**.

  - Phase 55 - **gravity in the no-water overall-composition model**:
    phase potentials now use `grad(p) - avg(rho_phase) * g * grad(z)` for oil
    and gas. `validate_phase55.py` runs a closed vertical oil/gas column
    (10 cells, 5×20 days, gravity on): pressure to **2.4e-4 bar**,
    composition to **1.6e-6**, saturation to **8.4e-6**, with component mass
    conserved and methane enriching upward.

  - Phase 56 - **first `model.water = true` slice** in
    `compositional/model.py`: water saturation is a primary variable; oil/gas
    saturations are scaled by `(1-sW)`; the water conservation equation and
    water flux are active for closed-domain cases. `validate_phase56.py` runs a
    no-well, no-gravity, no-capillary water/oil/gas case (10 cells, 5×20 days):
    pressure to **3.7e-4 bar**, water saturation to **1.6e-8**, composition to
    **7.5e-7**, oil/gas saturation to **3.2e-6**, with water and component mass
    conserved.

  - Phase 57 - **compositional water wells**: water-active states now carry
    `qWs` in addition to `qOs`, `qGs`, and `bhp`. The well model handles water
    perforation rates, water conservation sources, oil/gas component sources,
    and rate/BHP controls. `validate_phase57.py` runs a water injector and BHP
    producer (10 cells, 5×30 days): pressure to **1.1e-5 bar**, `sW` to
    **3.4e-7**, composition to **1.3e-6**, oil/gas saturation to **2.1e-6**,
    well rates to **1.5e-9 m³/s**, and BHP to **1.1e-5 bar**.

  - Phase 58 - **compositional capillary pressure**: optional `pc_ow` and
    `pc_og` functions split phase pressures in the flux potentials (`pW =
    p-pcOW`, `pG = p+pcOG`). `validate_phase58.py` runs a no-well,
    no-gravity water/oil/gas case with `pcOW(sW)=pc0*(1-sW)` (10 cells,
    5×20 days): pressure to **2.3e-4 bar**, `sW` to **1.3e-10**, composition
    to **5.8e-10**, and oil/gas saturation to **1.4e-6**.

  - Phase 59 - **`opt.pressure` pressure-equation assembly**: `assemble(...,
    pressure_reduction=True)` now computes MRST-style numerical partial-volume
    weights from the accumulation Jacobian (`getPartialVolumes` path) and
    collapses the conservation equations into one weighted pressure equation.
    `validate_phase59.py` compares the reduced pressure residual, weights, and
    pressure Jacobian block against MRST's `PressureOverallCompositionModel`.

  - Phase 60 - **pressure-only sequential Newton step**:
    `OverallCompositionModel.pressure_step()` assembles with full composition
    AD variables, keeps only pressure plus well variables active, and solves the
    reduced `opt.pressure` system while freezing transport variables.
    `validate_phase60.py` compares the no-well pressure-only step against a
    direct MRST `PressureOverallCompositionModel` Newton loop: pressure to
    **0.33 bar**, composition frozen exactly.

  - Phase 121 - **sequential transport**: the first no-water/no-well
    `transport_step()` is MATLAB-validated against MRST's
    `TransportOverallCompositionModel`. PyMRST now carries MRST-style frozen
    pressure total flux on the state and uses density-weighted component
    transport fluxes.
  - Phase 122 - **water sequential transport**: the water-active no-well
    `transport_step()` is MATLAB-validated against MRST's
    `TransportOverallCompositionModel`. The water equation now uses MRST's
    `rhoW*sW*sT` accumulation, `rhoW*sT` flux component, and `(1-sW)` oil/gas
    saturation scaling.
  - Phase 123 - **sequential well transport**: the no-water well case is
    MATLAB-validated against MRST's `TransportOverallCompositionModel`.
    PyMRST now stores the pressure step's frozen well streams on the state
    (`well_flux` reservoir phase flux and `well_components` component mass
    flux) and the transport half consumes them without solving well variables.
  - Phase 124 - **water sequential well transport**: the water-active well
    source path is MATLAB-validated against MRST's
    `TransportOverallCompositionModel`. PyMRST now mirrors MRST's iterative
    `wellSol.components` update inside transport assembly, which matters for
    the next Newton iteration in water/well cases.
  - Phase 125 - **sequential schedule wrapper**: `run_schedule` now drives
    `set_time_discretization('explicit')` through multiple report-step
    controls and isolates active well controls with deep copies, so switches
    during a step do not mutate the stored schedule or the model after the run.
  - Phase 126 - **facility group allocation modes**: compositional group
    controls now support current-rate, equal-share, and explicit-fraction
    allocation of a violated group limit, with active-well re-normalization and
    schedule-safe state-guess resets.
  - Phase 127 - **schedule-level facility blocks**: compositional
    `simple_schedule` now carries source, separator, and group-control blocks
    directly, and `run_schedule` deep-copies active facility controls before
    every report step so switching logic cannot mutate stored schedule controls
    or leak back into the model.
  - Phase 128 - **mixed facility-control schedule scope**: schedules that do
    not manage separator/group controls now preserve model-level defaults, while
    mixed schedules that do manage those fields clear stale facility blocks on
    report steps whose controls set them to `None`.
  - Phase 129 - **separator example packed-problem parity**:
    `book_ii/compositional_separator_example.py` now mirrors MRST's packed
    problem structure for `compositionalSeparatorExample.m`, creating
    `Overall-NoSep`, `Overall-MultiSep`, and `Overall-SingleSep` over one
    scaled SPE5 depletion schedule while leaving the heavy separator-coupled
    solve optional.
  - Phase 130 - **VLE intro example section parity**:
    `book_ii/intro_to_vapor_liquid_equilibrium.py` now mirrors MRST's
    Rachford-Rice, ternary Peng-Robinson, and SPE5 pressure/temperature flash
    sections as reusable data-producing functions with scaled default grids.
  - Phase 131 - **natural/overall single-cell example parity**:
    `book_ii/show_natural_overall.py` and `show_natural_overall2.py` now use
    MRST's single-cell boundary-driven setup instead of the older two-cell
    workaround, backed by an ADI zero-face broadcasting fix.
  - Phase 132 - **3D six-component example parity**:
    `compositional_example_3d_six_components.py` now mirrors MRST's
    `lumped_1` quarter five-spot setup with corner vertical wells, natural
    variables by default, pure-CO2 component streams, ramped two-year schedule,
    scaled defaults, and a full-size `20 x 20 x 2` option.
  - Phase 133 - **validation-simple packed output parity**:
    `compositional_validation_simple.py` now mirrors MRST's
    `packSimulationProblem` / `getPackedSimulatorOutput` structure, including
    explicit MRST/E300/AD-GPRS comparison slots and ternary displacement paths.
  - Phase 134 - **larger-problem tutorial timing parity**:
    `compositional_larger_problem_tutorial.py` now mirrors MRST's backend
    timing table structure, including assembly/linear/total timing columns,
    Sparse/Diagonal/MEX-row backend rows, Sparse+Direct solver timing, and
    MRST's timing-plot tick-label branch.
  - Phase 135 - **accuracy example plotting-data parity**:
    `book_ii/compositional_accuracy_example.py` now mirrors MRST's plotting
    sections by returning component profiles, gas saturation profiles,
    viscosity/density/property overlays, and cumulative implicit iteration
    tables from the packed Explicit/Natural/Overall runs.
  - Phase 136 - **miscibility example section-data parity**:
    `book_ii/compositional_miscibility_example.py` now mirrors MRST's result
    extraction sections by returning pressure-sweep gas-saturation front
    animation data, raw packed states/reports, and natural-vs-overall coarse
    timestep iteration tables.
  - Phase 137 - **SPE1 example plotting-data parity**:
    `run_compositional_spe1.py` now mirrors MRST's plotting sections by
    returning natural/overall/reference well-solution datasets, reference
    `PseudoOil`/`PseudoGas` component-rate fields, and cumulative nonlinear
    iteration tables.
  - Phase 138 - **hierarchical facility group controls**:
    compositional group limits can now reference direct wells, subgroup names,
    `group:<name>` entries, or nested `FacilityGroupControl` objects, and parent
    direct-member fractions are pushed down to active leaf wells.
  - Phase 139 - **deck-imported facility group trees**:
    MRST-shaped `GRUPTREE`, `WELSPECS`, `WGRUPCON`, `GCONPROD`, and `GCONINJE`
    schedule-control records now convert into native `FacilityGroupControl`
    trees.
  - Phase 140 - **compositional schedule report metadata**:
    compositional example schedules now return MRST-style report dictionaries
    with one-based controls, control-change flags, nonlinear iterations,
    timestep cuts, accepted ministeps, and convergence flags; packed-problem
    output helpers preserve those rich reports.
  - Phase 141 - **deck-imported compositional well controls**:
    MRST-shaped `WCONPROD`, `WCONINJE`, `WELTARG`, and `WELLTARG` records can
    now update existing native `Well` objects with status, sign, active
    controls, targets, limits, and injector phase streams.
  - Phase 142 - **deck-imported group economics/network metadata**:
    MRST-shaped `GECON` and `GRUPNET` rows are now preserved on imported
    `FacilityGroupControl.extra` metadata, including metadata-only child groups,
    so exact deck/example ports do not discard economics/network records before
    full enforcement is implemented.
  - Phase 143 - **MRST well-control keyword aliases**:
    the deck well-control updater now accepts `WCONHIST`, `WCONINJH`, and
    legacy `WCONINJ` aliases in addition to `WCONPROD`/`WCONINJE`, with
    `WELTARG`/`WELLTARG` target updates applied across all aliases.
  - Phase 144 - **deck-control schedule conversion**:
    MRST-like `schedule.control` dictionaries can now be converted directly
    into native `ScheduleControl` objects. The conversion applies deck well
    controls to base `Well` objects, imports group controls and metadata, and
    preserves PyMRST-native `bc`, `src`, and separator/group fields for
    report-step switching.
  - Phase 145 - **conservative `GECON` economics enforcement**:
    imported `GECON` metadata now participates in native facility updates for
    producer groups. Finite minimum oil/gas rates and maximum water-cut,
    gas-oil-ratio, and water-gas-ratio criteria shut in the active group wells
    and zero their state rate guesses; richer workover/follow-on actions remain
    metadata.
  - Phase 146 - **`GRUPNET` guide-rate allocation**:
    imported child-group `GRUPNET` guide rates with network use enabled now seed
    native parent-group allocation fractions when no explicit `WGRUPCON`
    fractions are present. The original `GRUPNET` rows remain available as
    metadata for later VFP/network enforcement.
  - Phase 147 - **boundary-control schedule conversion**:
    MRST-like boundary-control dictionaries now build native
    `BoundaryCondition` and `ScheduleControl` objects for report-step switching,
    including side-based and explicit-face flux/pressure records with changing
    compositional streams.
  - Phase 148 - **source-control schedule conversion**:
    MRST-like source-control dictionaries now build native `Source` and
    `ScheduleControl` objects for report-step switching, including explicit
    source/sink cell records with changing rates, saturations, and
    compositional injection streams.
  - Phase 149 - **mixed forcing-control schedule conversion**:
    MRST-like controls can now carry boundary-condition records and explicit
    source/sink records together in the same report-step control, with both
    converted into native `ScheduleControl` blocks.
  - Phase 150 - **exact example catalog parity**:
    the compositional example catalog now maps every MRST `.m` example,
    including `book-ii` and `utils`, to a concrete PyMRST module/file and
    validates that the mirror remains complete.
  - Phase 151 - **facility-switch report diagnostics**:
    group rate-limit switches, GECON economic shut-ins, and well-control
    switches are now recorded on the compositional model and surfaced through
    MRST-style schedule report dictionaries.
  - Phase 152 - **WGRUPCON status/metadata parity**:
    imported `WGRUPCON` guide-control rows are preserved as
    `FacilityGroupControl.extra["WGRUPCON"]`, inactive guide rows no longer seed
    explicit group-allocation fractions, and all-inactive guide sets fall back
    to current-rate allocation.
  - Phase 153 - **GECON end-run request reporting**:
    imported `GECON` rows with `end_run=YES` now set model end-run diagnostics
    when economics criteria are violated, expose `EndRunRequested` and
    `EndRunGroups` in schedule reports, and stop native schedule execution after
    the accepted report step by default.
  - Phase 154 - **WGRUPCON fifth-column multiplier parity**:
    the fifth numeric field from MRST `readWGrupCon` is now preserved as
    `guide_multiplier`, included in `effective_guide_rate`, and used when
    imported `WGRUPCON` rows seed explicit group-allocation fractions.
  - Phase 155 - **forcing-control composition aliases**:
    MRST-style switched boundary/source controls now accept `compi`, `zi`,
    `composition_i`, `comp_i`, and related aliases wherever PyMRST already
    accepted `components`/`z`, and those aliases are consumed instead of
    leaking into schedule-control metadata.
  - Phase 156 - **forcing-control boundary table records**:
    MRST-style switched boundary controls now accept one table-shaped `bc`
    record with parallel `face`, `type`, `value`, `sat`, and component rows,
    including mixed `flux`/`pressure` entries. This reduces the gap between
    PyMRST-native helper records and exact MATLAB `bc` struct ports.
  - Phase 157 - **GCONPROD/GCONINJE metadata preservation**:
    imported group producer/injector control rows now keep their raw MRST row,
    active control mode, status, guide-rate fields, exceed/procedure flags, and
    late target columns in `FacilityGroupControl.extra`, while preserving the
    existing native limit/status behavior.
  - Phase 158 - **GCONPROD guide-rate column parity**:
    positional MRST `GCONPROD` rows no longer treat column 9 as a native
    total-rate limit. That column is `guide_rate` in MRST `readGConProd` and is
    now metadata-only unless an explicit mapping alias such as `target` or
    `total_rate` requests a native group `rate` limit.
  - Phase 159 - **GCONINJE duplicate replacement parity**:
    imported `GCONINJE` rows now mirror MRST `readGConInje` replacement:
    duplicate `(group, phase)` rows keep the last record before native limits
    and metadata are formed, while distinct phase rows for the same group
    remain available.
  - Phase 160 - **GCONPROD/GCONINJE mapping raw-row order parity**:
    mapping-style imported group-control records now preserve metadata `raw`
    rows in MRST template column order instead of Python dictionary insertion
    order, while sequence records remain byte-for-byte in their original row
    order.
  - Phase 161 - **GECON/GRUPNET/WGRUPCON mapping raw-row order parity**:
    mapping-style economics, network, and well-guide control metadata now use
    MRST template column order for `raw` rows, so exact example ports can depend
    on deck keyword layout rather than Python dictionary insertion order.
  - Phase 162 - **GECON/GRUPNET duplicate replacement parity**:
    imported economics and network rows now mirror MRST keyword parsing by
    replacing earlier rows with the same group name before schedule/group
    metadata is built.
  - Phase 163 - **well-control keyword table replacement parity**:
    imported `WCONPROD`, `WCONHIST`, `WCONINJE`, `WCONINJ`, and `WCONINJH`
    tables now mirror MRST `assignControlRecords`: the last row for a well wins
    within one keyword, and a well assigned by one control keyword is removed
    from the other well-control keyword tables before `WELTARG` is applied.
  - Phase 164 - **whole-well WELOPEN status parity**:
    imported whole-well `WELOPEN` rows now open/shut the active
    post-replacement `WCON*` control row for matching wells before `WELTARG` and
    native `Well` construction. Completion-level `WELOPEN` rows remain a
    conservative no-op until completion-control data structures are ported.
  - Phase 165 - **completion-status support and wildcard WELOPEN ranges**:
    native `Well` objects now expose optional MRST-style `cstatus`; compositional
    well equations mask inactive completions by active `WI`; and wildcard-I/J/K
    `WELOPEN` rows with lower/upper perforation bounds can open or shut selected
    completions.
  - Phase 166 - **grid-aware positive-I/J/K WELOPEN matching**:
    when a Cartesian grid is supplied to `apply_deck_well_controls`, positive
    one-based `WELOPEN` I/J/K values now match well perforations through
    `cells.indexMap` if present, while zero I/J/K entries remain MRST-style
    wildcards.
  - Phase 167 - **schedule-level grid-aware WELOPEN forwarding**:
    deck-style schedule conversion helpers now accept `grid=G` and forward it to
    well-control conversion, so per-control completion shut/open status changes
    survive when building native `ScheduleControl` and `SimpleSchedule` objects.
  - Phase 168 - **WELSPECS/COMPDAT well-geometry import**:
    native compositional helpers can now expand parsed MRST `COMPDAT` rows into
    real PyMRST `Well` objects, including multi-row K ranges, explicit WI,
    radius/skin/direction metadata, WELSPECS reference depth/group metadata, raw
    row preservation, and initial completion `cstatus`. Full deck parser
    coverage for every rare COMPDAT column remains a deeper parity item.
  - Phase 169 - **schedule-level COMPDAT geometry bootstrapping**:
    deck-style schedule conversion can now start with `base_wells=None` and
    bootstrap native wells from the first `WELSPECS`/`COMPDAT` control block when
    supplied with `grid` and `rock`, then apply report-step `WCON*` and
    `WELOPEN` changes as isolated control copies.
  - Phase 170 - **COMPDAT KH and completion metadata parity**:
    defaulted connection transmissibility can now use COMPDAT `KH` through the
    existing Peaceman well-index calculation, mixed explicit/default WI rows keep
    supplied connection WI where present, and `SATNUM`, `KH`, and D-factor are
    retained per completion for exact example ports.
  - Phase 171 - **COMPDAT/WELOPEN completion status text parity**:
    imported completions now retain MRST status text as `cstatus_text` in
    addition to solver-facing boolean `cstatus`, so exact deck/example ports can
    distinguish `SHUT` from `STOP`; completion-level `WELOPEN` updates keep the
    text metadata and boolean activity synchronized on copied schedule wells.
  - Phase 172 - **default COMPDAT insertion for WELSPECS-only wells**:
    native well import now mirrors MRST `insertDefaultCOMPDAT` by creating
    vertical all-layer open completions for wells listed in `WELSPECS` but
    missing from `COMPDAT`, using the WELSPECS I/J location and MRST's default
    0.1 completion diameter.
  - Phase 173 - **overlapping COMPDAT last-record-wins parity**:
    expanded COMPDAT completion rows are collapsed by active cell before
    `add_well`, keeping the last record's WI, radius, status text, SATNUM, KH,
    skin, and D-factor metadata for repeated perforations, matching MRST
    `processWells` overlap semantics.
  - Phase 174 - **default WCON BHP target parity**:
    missing producer BHP columns in `WCONPROD` now default to one atmosphere,
    and missing injector BHP columns in `WCONINJE` now default to `6895*barsa`,
    while explicit infinite BHP limits remain unlimited, matching MRST
    `processWells` defaults.
  - Phase 175 - **legacy WCONINJ column-layout parity**:
    `WCONINJ` list rows now use the MRST legacy column layout for BHP/THP
    (columns 9/10) instead of being forced through the newer `WCONINJE` layout;
    the native compact analogue shape remains accepted for existing examples.
  - Phase 176 - **WCONHIST BHP column-layout parity**:
    full MRST-shaped `WCONHIST` BHP controls now read BHP from the
    history-control column 10, and `WELTARG` updates that slot, while
    compact native analogue rows and ordinary `WCONPROD` parsing remain
    unchanged.
  - Phase 177 - **WCONHIST historical LRAT target parity**:
    full MRST-shaped `WCONHIST` `LRAT` controls now use the historical
    oil-plus-water target (`-(ORAT + WRAT)`) from columns 4/5 rather than
    the ordinary `WCONPROD` LRAT column; compact native analogue rows remain
    accepted.
  - Phase 178 - **WCONINJH historical rate-control parity**:
    full MRST-shaped `WCONINJH` rows are now always rate controlled, using
    column 4 as the active historical rate and `6895*barsa` as the default BHP
    limit; `WELTARG RATE` updates that history-rate slot while older textual
    native analogue rows remain accepted.
  - Phase 179 - **WCONHIST RESV_HISTORY parity**:
    full MRST-shaped `WCONHIST RESV` rows now import as `resv_history`, using
    the negative sum of historical ORAT/WRAT/GRAT columns for the active target
    and MRST's WOG phase split for the well composition; ordinary `WCONPROD`
    `RESV` rows still map to native `vrat`.
  - Phase 180 - **WCONHIST historical phase-composition parity**:
    full MRST-shaped `WCONHIST` rows now update producer composition vectors
    for `ORAT`, `WRAT`, `GRAT`, `LRAT`, and `BHP` using the same WOG phase
    splits as MRST `process_wconhist`; compact native analogue rows keep their
    prior compatibility behavior.
  - Phase 181 - **WCONPROD phase-composition parity**:
    ordinary producer controls now import MRST `process_wconprod` composition
    vectors for `ORAT`, `WRAT`, `GRAT`, `LRAT`, `RESV`, and `BHP`; `RESV`
    continues to map to PyMRST's native `vrat` control, but its composition is
    now the MRST all-phase producer stream rather than the previous well value.
  - Phase 182 - **WCON VFP-index metadata parity**:
    `Well.vfp_index` now preserves MRST's deck table selector from `WCONPROD`
    column 11, legacy `WCONINJ` column 11, and `WCONINJE` column 9; MATLAB well
    structs imported through `mrst_mat` also retain the field for native
    `VFPTable` binding.
  - Phase 183 - **VFP-index table binding parity**:
    positive `Well.vfp_index` values now bind to one-based injector/producer
    `VFPTable` lists in both generic `FacilityModel` and native compositional
    model setup, and compositional `thp`/`vfp` controls evaluate the bound table
    BHP from current oil/gas/water surface-rate ratios.
  - Phase 184 - **Deck VFP table schedule parity**:
    MRST-like `VFPINJ` and `VFPPROD` records on schedule controls now convert to
    native `VFPTable` objects, preserve sparse one-based table numbering, and
    bind through `run_schedule` when control wells select a `vfp_index`.
  - Phase 185 - **`selectFacilityFromDeck` layout parity**:
    PyMRST now has a native facility selector at the MRST-matching package
    location. It reuses or creates `FacilityModel`, collects deck VFP tables
    across schedule controls, preserves one-based sparse table slots, warns on
    overwrites, and rebinds wells with imported `vfp_index` selectors.
  - Phase 186 - **separator intermediate stream diagnostics parity**:
    `SeparatorGroup.compute_surface_mole_rates` can now return the same
    per-stage `molestreams`, `densities`, and `molefractions` diagnostics that
    MRST exposes as the second output of `computeSurfaceMoleRates`; tree
    `compute_surface_rates` results also retain those diagnostics.
  - Phase 187 - **separator example diagnostics parity**:
    the MRST-layout `compositional_separator_example` now returns
    `tree_separator_streams` directly beside the final surface mole streams, so
    example consumers can inspect the same intermediate separator stream data
    that MRST scripts obtain from `computeSurfaceMoleRates`.
  - Phase 188 - **whole MRST schedule mapping conversion**:
    exact example ports can now hand a full MRST-like schedule mapping
    (`step.val`, one-based `step.control`, and `control`) to
    `simple_schedule_from_mrst_schedule`; the helper delegates to the existing
    deck, boundary, source, VFP and facility converters and preserves model
    forcing restoration through `run_schedule`.
  - Phase 189 - **`GCONPROD` shut-style exceed actions**:
    producer group rate-limit violations now honor imported `GCONPROD`
    `exceed_action` values `SHUT`/`STOP` by shutting active producer wells,
    zeroing their state rate guesses, and carrying the action through schedule
    report diagnostics; `NONE`/default actions keep the existing target
    redistribution behavior.
  - Phase 190 - **`GCONINJE` injection-limit diagnostics**:
    injector group target redistributions now mark the matching imported
    `GCONINJE` metadata row with the last violation/action and carry the MRST
    keyword, phase, and control mode through schedule report diagnostics.
  - Phase 191 - **`GCONPROD` redistribution diagnostics**:
    ordinary non-shut producer group redistributions now also mark the matching
    imported `GCONPROD` metadata row with the last violation/action/exceed
    context while leaving shut-style exceed actions untouched.
  - Phase 192 - **MATLAB-struct schedule ingestion**:
    `simple_schedule_from_mrst_schedule` now accepts MATLAB/scipy-style struct
    schedules and object-array controls directly, including nested `BC`, `SRC`,
    deck/facility records, and control metadata, so exact example ports do not
    need a manual dictionary-cleanup layer before entering PyMRST.
  - Phase 193 - **MATLAB-struct facility deck ingestion**:
    lower-level compositional facility converters now accept MATLAB/scipy-style
    object records directly for WELSPECS/COMPDAT well import, WCON*/WELTARG
    well-control updates, GCON*/GECON/GRUPNET/WGRUPCON group conversion, and
    whole schedule facility conversion, closing the remaining dictionary-only
    path beneath the exact schedule helper.
  - Phase 194 - **GECON runtime diagnostic context**:
    conservative economics shut-ins now stamp imported `GECON` metadata with
    the last action and affected wells, and schedule reports carry the MRST
    keyword, workover procedure, follow-on value, and end-run request that
    triggered the economics switch.
  - Phase 195 - **GRUPNET allocation diagnostic provenance**:
    parent groups whose explicit fractions were seeded from child `GRUPNET`
    network guide rates now retain `GRUPNET_ALLOCATION` metadata, and runtime
    group-limit switch reports distinguish the violated constraint keyword
    (`GCONPROD`/`GCONINJE`) from the `GRUPNET` allocation source.
  - Phase 196 - **GCONPROD NRAT native group limit**:
    imported `GCONPROD` network-rate (`NRAT`) columns now become conservative
    native `nrat` group limits. Producer groups enforce them against total
    surface rate, diagnostics stamp the matching metadata row, and switched
    member wells close with total-rate `nrat` control equations.
  - Phase 197 - **GCONPROD RESV native group limit**:
    imported `GCONPROD` reservoir-rate (`RESV`) columns now become native
    `vrat` group limits. Positional and mapping rows both enforce total
    surface-rate redistribution and stamp the matching imported metadata row.
  - Phase 198 - **GCONPROD late-column diagnostics**:
    late `GCONPROD` columns that still do not have a safe native enforcement
    path - damping, free-group-control, region, guide context, voidage,
    wet-gas, calorific/CRAT, TRGP, and procedure - now survive as named
    metadata and runtime report context rather than only raw-row slots.
  - Phase 199 - **GCONINJE late-column diagnostics**:
    injection-side late columns - reinjection target, voidage target, guide
    rate/phase, TRGP, VREP, and wet-gas - now reach runtime diagnostics for the
    selected imported `GCONINJE` row while remaining metadata unless a native
    enforcement path is implemented.
  - Phase 200 - **WGRUPCON allocation diagnostics**:
    `WGRUPCON` guide rows that seed native group allocation fractions now have
    explicit allocation-provenance metadata and runtime report context, including
    active/inactive guide rows, guide multipliers, effective guide rates, and
    the selected allocation fractions.
  - Phase 201 - **native allocation diagnostics**:
    group limits that use PyMRST's built-in current-rate, equal, or explicit
    fraction allocation now report `NATIVE` allocation provenance with active
    wells, normalized weights, and raw fractions where present. This closes the
    diagnostic gap left after `GRUPNET` and `WGRUPCON` provenance were added;
    it intentionally does not change the allocation math.
  - Phase 202 - **facility-block change diagnostics**:
    compositional schedule reports now expose `SeparatorChanged` and
    `GroupControlsChanged` flags when one-based control selection changes
    managed separator/group blocks, and include those control-block changes in
    `FacilityChanged`. Schedules that do not manage facility blocks still leave
    model-level defaults untouched and unreported.
  - Phase 203 - **well/VFP facility-block change diagnostics**:
    compositional schedule reports now expose `WellControlsChanged`,
    `VFPInjectorTablesChanged`, `VFPProducerTablesChanged`, and
    `VFPTablesChanged` when one-based control selection changes managed well or
    VFP table blocks. These report-only flags make schedule-level facility
    input changes visible even when no nonlinear control switch occurs.
  - Phase 204 - **forcing-block change diagnostics**:
    compositional schedule reports now expose `BoundaryChanged`,
    `SourceChanged`, and `ForcingChanged` when one-based control selection
    changes managed boundary/source blocks. These report-only flags make
    switched forcing inputs inspectable without classifying them as facility
    changes.
  - Phase 205 - **fixed-step forcing report metadata**:
    the simpler `run_steps` helper now emits the same forcing-change fields as
    `run_schedule`; fixed boundary conditions are reported as first-step forcing
    inputs.
  - Phase 206 - **fixed-step source forcing support**:
    `run_steps` now also accepts fixed source terms, forwards them to the
    compositional model step, and reports them through the same
    `SourceChanged`/`ForcingChanged` metadata used by schedule controls.
  - Phase 207 - **fixed-step report schema parity**:
    `run_steps` now emits false/empty defaults for the same facility, VFP,
    switch, and end-run report keys as `run_schedule`, making fixed-step and
    scheduled compositional reports easier to consume uniformly.
  - Phase 208 - **packed fallback report schema parity**:
    `get_multiple_packed_simulator_outputs` now expands legacy results with
    only iteration vectors into the same MRST-like report key set, so packed
    example consumers no longer need a separate bare-report path.
  - Phase 209 - **packed fallback schedule metadata parity**:
    synthetic packed reports now recover `schedule.step.val/control` where
    available, including timesteps, singleton mini-steps, one-based controls,
    and control-change flags; only mismatched legacy rows remain unknown.
  - Phase 210 - **packed fallback schedule block-change parity**:
    synthetic packed reports now infer schedule-managed forcing and facility
    block changes from packed controls, keeping boundary/source changes separate
    from well, separator, group-control, and VFP facility changes.
  - Phase 211 - **boundary forcing alias parity**:
    boundary-control records now accept MRST-style pressure and flux aliases
    (`pres`/`p`/`bhp`/`dirichlet`, `rate`/`q`/`neumann`/`massrate`) across side
    records, sequence records, face tables, and forcing-control metadata
    filtering.
  - Phase 212 - **source forcing rate-alias parity**:
    source-control records now accept MRST-style `q`, `massrate`, and
    `mass_rate` aliases in direct, nested, source-schedule, and mixed forcing
    controls, while keeping source-only records out of the boundary converter.
  - Phase 213 - **case-insensitive forcing metadata cleanup**:
    boundary, source, and mixed forcing schedule converters now consume parsed
    MATLAB-style aliases case-insensitively (`Q`, `MassRate`, `COMPI`, `ZI`,
    etc.) rather than leaking them into `ScheduleControl.extra`.
  - Phase 214 - **case-insensitive forcing container keys**:
    nested boundary/source blocks can now arrive under MATLAB-style mixed-case
    names such as `BoundaryConditions`, `BCRecords`, `SourceTerms`, and
    `SrcRecords`; parsed container fields are consumed instead of being kept as
    schedule metadata.
  - Phase 215 - **case-insensitive native forcing passthrough**:
    already-built PyMRST `BoundaryCondition` and `Source` objects now pass
    through boundary, source, and mixed forcing controls under mixed-case aliases
    such as `Bc`, `BoundaryConditions`, `Src`, and `SourceTerms`, with defensive
    deep copies.
  - Phase 216 - **case-insensitive schedule side facility blocks**:
    simple, MRST-shaped, boundary, source, and mixed forcing schedules now
    consume `Wells`/`WELLS`, `SeparatorGroup`/`separatorGroup`, and
    `GroupControls`/`GROUPCONTROLS` aliases as native side blocks instead of
    preserving them as extra metadata.
  - Phase 217 - **native forcing aliases in generic/deck schedules**:
    `simple_schedule`, `schedule_controls_from_deck_controls`, and
    `simple_schedule_from_deck_controls` now deep-copy and consume native
    `BoundaryCondition`/`Source` objects under mixed-case forcing aliases such
    as `BoundaryConditions`, `BCRecords`, `SourceTerms`, and `SrcRecords`.
  - Phase 218 - **case-insensitive generic/deck schedule metadata cleanup**:
    the generic/deck schedule paths now filter consumed aliases
    case-insensitively, so uppercase MRST-style fields such as `WELLS`,
    `BOUNDARYCONDITIONS`, `SOURCETERMS`, `SEPARATORGROUP`, `GROUPCONTROLS`,
    `VFPINJ`, and `VFPPROD` do not leak into `ScheduleControl.extra`.
  - Phase 219 - **singular forcing container aliases**:
    direct, schedule, mixed forcing, generic, and deck schedule paths now accept
    singular/alternate forcing containers such as `Boundary`,
    `BoundaryCondition`, `SourceTerm`, and `SrcTerms`, including native
    `BoundaryCondition`/`Source` passthrough and metadata cleanup.
  - Phase 220 - **MATLAB cell-array forcing record flattening**:
    boundary/source forcing records loaded as `1 x N` or `N x 1` object arrays
    from `.mat` files are now flattened into individual MRST records, while
    numeric object arrays remain table rows for compact source/BC inputs.
  - Phase 221 - **forcing record field-name aliases**:
    boundary/source forcing records now accept plural and index-style aliases
    such as `values`, `vals`, `faceIndex`, `faceIndices`, `cellIndex`,
    `cellIndices`, `saturations`, and `S`/`s`, with consumed aliases filtered
    from schedule metadata.
  - Phase 222 - **boundary side-name aliases**:
    boundary controls now accept `faceSide`, `face_side`, `sideName`, and
    `side_name` wherever `side` was accepted, and these aliases are consumed
    from schedule metadata.
  - Phase 223 - **typed boundary side aliases**:
    boundary controls now accept `pressureSide`/`pressure_side` and
    `fluxSide`/`flux_side`, inferring the boundary type from the field name and
    consuming those aliases from schedule metadata. Explicit conflicting
    boundary types are rejected.
  - Phase 224 - **MRST `pside` boundary alias**:
    boundary controls now accept `pside`, `pSide`, and `PSide` as pressure-side
    aliases, matching the MRST helper name used by exact boundary-condition
    examples and MATLAB-style struct dumps.
  - Phase 227 - **MATLAB-backed `psideh` regression**:
    `dump_phase227_reference.m` now records MRST's core hydrostatic side
    pressure helper on a small Cartesian grid, and `validate_phase227.py`
    compares PyMRST against the saved faces, centroids, saturations, type
    labels, mobility-weighted density, gravity vector, and pressure values.
  - Phase 228 - **boundary-condition example plotting sections**:
    `compositional_boundary_conditions_example.py` now exposes the pressure,
    gas-saturation, CO2-composition, and nonlinear-iteration plot data sections
    from `compositionalBoundaryConditionsExample.m` as arrays/dictionaries
    instead of leaving them implicit in the returned state histories.
  - Phase 229 - **3D six-component example plotting frames**:
    `compositional_example_3d_six_components.py` now exposes MRST's per-report
    plotting loop for `compositionalExample3DSixComponents.m`: six component
    maps, pressure, `sO`, `sG`, view/subplot metadata, and cumulative report
    times. The older final-state `extract_plot_fields` helper remains for
    compatibility.
  - Phase 230 - **VLE intro SPE5 flash diagnostics**:
    `intro_to_vapor_liquid_equilibrium.py` now exposes the later diagnostic
    sections from `introToVaporLiquidEquilibrium.m`: SPE5 SSI iteration map,
    active-cell convergence history, and a stability/flash timing table. The
    MRST SSI+Newton/MEX branch remains an explicit unavailable slot until
    PyMRST has the corresponding Newton flash backend.
  - Phase 231 - **performance example timing-table parity**:
    `compositional_performance_example.py` now exposes the exact four-row
    timing matrix plotted by `compositionalPerformanceExample.m` - Sparse,
    Diagonal, Diagonal (MEX), and Diagonal (MEX+RowMajor) with assembly,
    linear-solve, and total time-per-iteration columns. PyMRST's extra
    pure-Python `Diagonal (RowMajor)` diagnostic row remains available in the
    raw benchmark output, but is intentionally omitted from the MRST-shaped
    plot table.
  - Phase 232 - **larger-problem tutorial plotting sections**:
    `compositional_larger_problem_tutorial.py` now exposes the final plot
    sections from `compositionalLargerProblemTutorial.m`: timing bar chart data
    with `figure(1 + useNatural)`, tick labels, legend/location metadata, and
    the assembly-time zoom vector and y-limit. This keeps exact example
    consumers from reconstructing MATLAB plotting semantics by hand.
  - Phase 233 - **showNaturalOverall2 runtime plotting section**:
    `show_natural_overall2.py` now records wall time for the natural-variable
    and overall-composition solves and exposes the final MRST bar-plot payload:
    a `1x2` subplot layout, `Natural`/`Overall` tick labels, and separate
    `Iterations` and `Wall time` bars.
  - Phase 234 - **showNaturalOverall convergence plotting sections**:
    `show_natural_overall.py` now exposes the two convergence figures from
    `showNaturalOverall.m`: composition-pressure and saturation-pressure paths
    over a flash-derived saturation contour map, with MRST legend, labels,
    solution marker, and `maxChange` style metadata. The current payload uses
    the solver's available initial-to-final paths; full nonlinear sub-iteration
    histories remain unavailable until the compact PyMRST timestepper exposes
    report-level intermediate states.
  - Phase 235 - **accuracy example plot-section metadata**:
    `compositional_accuracy_example.py` now wraps its existing component,
    gas-saturation, property, and cumulative-iteration arrays in MRST-shaped
    plot-section dictionaries with legend placement, y-limits, labels,
    dual-axis property metadata, and the cumulative-iteration title.
  - Phase 236 - **miscibility example plot-section metadata**:
    `compositional_miscibility_example.py` now exposes MRST's gas-front
    animation metadata (`figure(1)`, `summer(np)`, linewidth, axes labels, and
    per-pressure profiles) and the final formulation-iteration bar metadata.
    The latter carries an explicit unavailable reason when the scaled run skips
    the optional natural/overall comparison branch.
  - Phase 237 - **SPE1 example plot-section metadata**:
    `run_compositional_spe1.py` now exposes the interactive plotting sections
    from `runCompositionalSPE1.m`: Natural/Overall/Black-oil `plotToolbar`
    metadata with MRST titles, tight axes and `(30,30)` view, the
    `plotWellSols` dataset-name bundle, and cumulative nonlinear-iteration
    stairs metadata.
  - Phase 238 - **setupSimpleCompositionalExample schedule/reference output**:
    `setup_simple_compositional_example.py` now returns a compact MRST-shaped
    schedule object, clears well limits, assigns the same injected component
    stream and two-phase `compi` vector to all schedule wells, and exposes the
    optional `comparison.mat` fourth output as explicit unavailable metadata.
  - Phase 239 - **validation-simple plot-section metadata**:
    `compositional_validation_simple.py` now exposes MRST's component
    comparison, pressure/saturation comparison, interactive toolbar, and ternary
    diagram metadata, with unavailable E300/AD-GPRS references represented
    explicitly.
  - Phase 240 - **Contents.m catalog metadata**:
    `contents.py` now carries MRST's exact top-level `Contents.m` example list
    in order and with the MATLAB descriptions, separate from the expanded
    PyMRST catalog that also includes book-II examples and utility ports.
  - Phase 241 - **3D six-component result-plot metadata**:
    `compositional_example_3d_six_components.py` now exposes MRST's animation
    and `plotToolbar` sections: figure offset, subplot layout, component
    caxis/title metadata, pressure/sO/sG tail plots, drawnow frames, and final
    tight-axis toolbar metadata.
  - Phase 242 - **boundary-condition full plot metadata**:
    `compositional_boundary_conditions_example.py` now exposes the MRST
    permeability figure, Natural/Overall toolbar calls, 3-by-2 comparison
    animation styling, and `-o` nonlinear-iteration plot labels.
  - Phase 243 - **property Contents.m catalog metadata**:
    `properties/contents.py` and `properties/components/contents.py` now expose
    MRST's exact property catalog entries in order. The
    `CompositionalFlowPropertyFunctions` row is preserved as an explicit
    `missing_in_mrst_tree` entry because it appears in MRST's `Contents.m` but
    no same-named MRST 2024a file is present.
  - Phase 244 - **EquationOfStateComponent method-name parity**:
    `EquationOfStateComponent` now exposes MRST's `getPhaseComposition`,
    `getComponentDensity`, and `getPhaseComponentFractionInjection` camelCase
    APIs in addition to the Pythonic helpers, including model/getProps and
    injection-force call patterns.
  - Phase 245 - **separator facility Contents.m catalog metadata**:
    `ad_core.models.facilities.separators.contents` now mirrors MRST's
    separator `Contents.m` entries (`BasicSeparator`, `EOSSeparator`,
    `SeparatorGroup`) with explicit implementation metadata and importable
    PyMRST modules.
  - Phase 246 - **separator multi-output method parity**:
    separator result dataclasses now support MRST-style tuple unpacking for
    `separateComponentMassStream`, `separateComponentMoleStream`,
    `computeSurfaceRates`, and separator-stream diagnostics while preserving
    the named Python result fields.
  - Phase 247 - **shared separator allocation diagnostics**:
    `CompositionalState.separator_group_diagnostics` now records how a combined
    producer component stream was split back over wells after passing through a
    shared `SeparatorGroup`: producer order/names, per-producer streams, total
    separator stream, allocation shares, positive group surface rates, and the
    signed allocated oil/gas rates written to `qOs`/`qGs`.
  - Phase 248 - **separator example allocation diagnostics**:
    `evaluate_separator_example_allocation_diagnostics` now returns final
    separator allocation diagnostics aligned with MRST's no-separator,
    multi-stage separator, and single-separator packed-problem order, so the
    example exposes facility allocation behavior as well as separator flash-tree
    streams without forcing the optional separator-coupled Newton run.
  - Phase 249 - **separator allocation diagnostics in schedule reports**:
    compositional schedule reports now include
    `SeparatorAllocationDiagnostics` as a deep-copied payload whenever the
    accepted state has `separator_group_diagnostics`, making facility allocation
    behavior inspectable through `run_steps`, `run_schedule`, and packed-report
    passthroughs.
  - Phase 250 - **compact separator allocation report summary**:
    schedule reports now expose `SeparatorAllocationSummary` beside the full
    diagnostics payload, with producer order/names, shares, signed allocated
    surface rates, positive total surface rates, and the total component mass
    stream in report-friendly form.
  - Phase 251 - **SeparatorGroup property/method alias parity**:
    `SeparatorGroup` now accepts MRST's `phaseDestination` constructor keyword
    and exposes `phaseDestination`, `topologicalOrder`, `surfaceSeparator`, and
    `getTopologicalSort` aliases alongside the existing Pythonic attributes and
    methods.
  - Phase 252 - **separator constructor/EOSModel parity**:
    `BasicSeparator()` and `EOSSeparator()` now support MRST-style empty
    construction, and `EOSSeparator` stores the MRST `EOSModel` property with a
    Pythonic `eos_model` alias that is preserved by `clone`.
  - Phase 253 - **EOSSeparator local EOSModel mixture override**:
    `EOSSeparator.separateComponentMoleStream` now uses the separator's own
    `EOSModel` mixture when present before falling back to the incoming
    model/mixture argument, matching MRST's separator-local EOS override.
  - Phase 254 - **EquationOfStateComponent property alias parity**:
    component objects now expose synchronized MRST-style aliases
    `componentIndex`, `surfacePhaseMassFractions`, `surfacePhaseDensityPure`,
    and `molarMass` alongside the Pythonic dataclass fields.
  - Phase 255 - **CompositionalMixture method/property alias parity**:
    `Mixture` now exposes MRST-style `getNumberOfComponents`,
    `getBinaryInteraction`, `setBinaryInteraction`, and a setter-backed
    `acentricFactors` alias alongside the Pythonic methods used internally.
  - Phase 256 - **CompositionalPropertyModel class parity**:
    the MRST-layout fluids package now exports `CompositionalPropertyModel`
    with `computeMolarDensity`, `computeDensity`, `computeViscosity`,
    `computePseudoCriticalPhaseProperties`, and synchronized
    `volumeShift`/`volume_shift` properties over the native PyMRST formulas.
  - Phase 257 - **EquationOfStateModel object parity**:
    the MRST-layout EOS package now exports `EquationOfStateModel` with
    `CompositionalMixture`, `PropertyModel`, PR/SRK/RK/PRCORR `setType`
    constants, `shortname`, component-name/count helpers, and cubic
    coefficient/root helpers.
  - Phase 258 - **fraction utility parity**:
    the MRST-layout `utils` package now exposes `ensureMinimumFraction`,
    `getNonUnitMassFraction`, and `getNonUnitMoleFraction` plus Pythonic
    aliases, matching MRST's matrix and cell-array style conversions.
  - Phase 259 - **simple-water component utility parity**:
    `getComponentsTwoPhaseSimpleWater` now mirrors MRST's per-component
    phase-mass cell-array layout for water-active and no-water compositional
    paths.
  - Phase 260 - **mass-format utility parity**:
    `formatMassString` now mirrors MRST's mass display formatter, including
    sign handling, unit order, pluralization, and depth-limited decimal output.

  **Still missing for full compositional parity:** MATLAB-backed validation
  for richer sequential pressure/transport schedules beyond report metadata,
  additional rare boundary/source schedule edge cases beyond the native conversion helper,
  and deeper MRST facility controls such as separator-controlled well equations
  and full enforcement of deck economics/network facility keywords beyond the
  conservative `GECON` shut-in, `GCONPROD` shut/redistribution traceability,
  `GCONINJE` diagnostic traceability, conservative `GCONPROD NRAT`/`RESV`,
  late-column `GCONPROD`/`GCONINJE` diagnostics, `WGRUPCON` allocation
  provenance, native allocation tracing, schedule facility-block diagnostics
  for separator/group/well/VFP inputs, schedule forcing-block diagnostics for
  boundary/source inputs, and `GRUPNET`
  guide-allocation/provenance subset.
- **`ad_core`** - ~~TPFA operators and a schedule driver, but not the
  state-function / model-class machinery MRST's AD-OO is built on.~~ The
  graph landed in Phase 24 (state functions, groupings, the PVT/flow/flux
  evaluator sets, `PhysicalModel`/`ReservoirModel`, the facility, and a real
  `NonLinearSolver`), carrying black-oil end to end.

  It is still **14 modules and 22 state functions against MRST's 345 files**,
  and the earlier wording here ("still thin") was too vague to be useful.
  What is actually missing, by consequence:

  *Physics you cannot currently model*
  - ~~**Boundary conditions and sources for AD models**~~ - done (Phase 41).
    This was missing from earlier versions of this list, which was an omission:
    the AD models could be driven by **wells and nothing else**, so every case
    had to be a closed box with a well in it. `boundary_conditions.py` ports
    `addFluxesFromSourcesAndBC` / `getBoundaryConditionFluxesAD` /
    `getSourceFluxesAD` - pressure and flux boundaries, and sources - and
    matches MATLAB at machine precision (1.5e-16) on three well-less cases.
    (No `resflux` boundaries; those need the sequential module.)
  - ~~**Capillary pressure**~~ - done (Phase 38): `CapillaryPressure` and
    `PhasePressures`, with `PressureGradient` now returning one gradient *per
    phase*. Matches MATLAB to 1.3e-12, and is inert when no `pcOW` is given, so
    Phases 5/5b/24/27 are undisturbed. The `ad-blackoil` layout pass added
    SWOF/SGOF table support for `pcOW` and `pcOG` (`pW = pO - pcOW`,
    `pG = pO + pcOG`, with `pcWG = pcOW + pcOG`). J-function scaling is
    available through `apply_j_function_scaling` / `init_deck_blackoil_fluid`
    with rock. A Killough gas-scanning hysteresis helper is present. Still
    missing: full deck-driven capillary hysteresis integration.
  - ~~**Well control switching**~~ - done (Phase 39):
    `FacilityModel.update_switched_controls` (MRST's
    `updateSwitchedWellControls`), with `W.lims` and the `bhp`/`rate`/`orat`/
    `wrat`/`grat`/`lrat`/`vrat` control equations a well can switch into.
    Matches MATLAB to 2e-12, control mode by control mode. (No group controls,
    no `allowWellSignChange`, no well status/shut-in logic.)
  - ~~**Equilibration**~~ - done (Phase 42): `initialization.py` ports
    `initializeEquilibriumPressures` / `initializeEquilibriumSaturations` /
    `initStateBlackOilAD` - each phase integrated along its own hydrostatic
    gradient from its contact, saturations by inverting the capillary curve.
    Matches MATLAB at machine precision on both a sharp contact and a capillary
    transition zone, and the resulting state is *exactly* at rest (300 days,
    zero change). Still missing: multiple equilibration regions, `rs`/`rv` vs
    depth tables, and deck-driven `EQUIL` parsing.
  - ~~**Vaporised oil** (`RvMax`, `rv`) - black-oil is disgas-only, no wet
    gas.~~ **Done** (Phase 47): `OilComponent`, the three-way status switch
    (status 2 = gas with no free oil, `rv` primary), the full flash, `rv` in the
    density, and the well's dissolution matrix `d = 1 - rs*rv`. Matches MATLAB
    at machine precision on a gas-condensate column, statuses identical.
    **Black-oil is complete.**
  - ~~**Aquifers** (`AquiferModel`)~~ - done (Phase 48):
    `ad_core/aquifer.py` ports MRST's Fetkovich analytic aquifer -
    `computeAquiferFluxes` (the `alpha J (p_aq - p_w + rho_w g dz) coef`
    influx with the `coef = (1-exp(-dt/Tc))/(dt/Tc)` relaxation),
    `addAquifersContribution`, and the `updateAfterConvergence` pressure/volume
    drawdown - wired into `GenericBlackOilModel` (a 1-line water-equation
    contribution) and `NonLinearSolver` (a guarded post-step hook). The aquifer
    pressure/volume ride on the state, so timestep cuts restart correctly. In
    PyMRST's surface-volume equations the influx enters with scale 1 where MRST
    (mass equations) uses `rhoWS`. In-file self-tests cover the flux, the
    coefficient limits, the gravity head, the least-squares equilibrium init,
    the drawdown update, two-aquifer routing, and an end-to-end sealed-reservoir
    drive.

    Validated against MRST **twice**, because the formulas and the coupling are
    different things and only one of them is arithmetic:
    `validate_phase48.py` cross-checks the formulas on fixed inputs (MRST's own
    connection pressures and shrinkage factors), matching **bit for bit**; and
    `validate_phase48b.py` runs a *coupled* simulation - a producer pulling a
    field down, an aquifer holding it up over twenty steps - which is the only
    thing that tests the sign of the water-equation contribution and the timing
    of the post-step update. Pressures match to 4.4e-11, the aquifer's own
    volume to 2.7e-15. (Run `dump_phase48_reference.m` and
    `dump_phase48b_reference.m` in MATLAB first.)

    ECLIPSE `AQUANCON`/`AQUFETP` deck parsing (`processAquifer`) is now
    ported under `deckformat/params/rock`, with `select_aquifer_from_deck`
    building the existing `AquiferModel`. `CarterTracyAquiferModel` adds a
    convolution-history aquifer with the same contribution/update interface.
    MRST 2024a in this tree does not include a Carter-Tracy aquifer class, so
    this path is self-tested rather than MATLAB-reference validated.
    *MRST finding:* `computeInitAquifer.m` references an undefined `V_aq` on
    its last line and cannot run in 2024a.
  - ~~VFP tables and multisegment topology/equations~~ - `VFPTable`,
    `MultisegmentWell`, `well_bore_friction`, and
    `assemble_multisegment_equations` exist, with explicit THP/VFP facility
    closure. `GenericBlackOilModel` now automatically packs MS well extra
    primary variables into the black-oil Newton system (`validate_phase109.py`).
  - **Hysteresis** in full relative permeability / capillary-pressure state
    integration. Killough gas relative-permeability scanning is available as a
    helper.

  *The gap that gated the roadmap*
  - ~~**The component framework**~~ - done (Phase 40): `Component`,
    `ImmiscibleComponent`, `GasComponent` (dissolved gas as a component in two
    phases) and `ConcentrationComponent`. The conservation equations now loop
    over `model.components`, not over phases, so `n_components != n_phases` is
    routine. Proved by adding a passive tracer as a new component with **no
    change to the flow model** (`ad_blackoil/models/tracer_black_oil_model.py`); black-oil is
    reproduced to machine precision and all six MATLAB validations still pass.
    Still missing from MRST's component surface: `ComponentMobility` /
    `ComponentPhaseDensity` as separate state functions, the component-aware
    *facility* state functions (`WellComponentPhaseFlux` and friends - PyMRST's
    well model still handles components inline), and `getPhaseComposition` /
    `getInjectionMassFraction` for compositional.

  *Scale, not capability*
  - `solvers/` (15 files): **CPR is done** (Phase 43) - AMG on the decoupled
    pressure system, ILU on the rest, GMRES around the pair, plus
    `select_linear_solver` (MRST's `selectLinearSolverAD`). The iteration count
    is flat across a 200x range in problem size, which is what lifts the field
    ceiling: 432,000 cells now solve where the direct solve could not go, and
    98,000 cells go 3.3x faster. Below ~30k unknowns the *direct* solve is still
    faster and the selector says so.
    Still missing: multiscale (`msrsb`, `msfvm`), `domain-decomposition`, and
    the block-preconditioner variants of CPR (`AMGCL`/`AMGCL_CPR` bindings).
  - `backends/` (47 files): one AD backend (sparse `ADI`). No
    `DiagonalAutoDiffBackend` / `GenericAD` (block-diagonal Jacobians); ~20 of
    those files are MEX kernels.
    **Deliberately not ported, and this is a measurement rather than a
    judgement.** It was next on the list as "a large constant-factor win" until
    it was profiled: with CPR in place, assembly is **2-9%** of a Newton
    iteration (0.13 s against 5.48 s at 98,000 cells). Making it *free* buys 2%
    at field scale. The 91% was the ILU inside CPR, and that is what got fixed
    instead (Phase 45); after that the AMG application was 80%, and that got
    fixed too (Phase 46). Worth revisiting only if the linear solve stops
    dominating - re-profile before starting, do not take this note's word for
    it.
  - ~~`timesteps/` (6 files): no timestep selectors - the solver *cuts* on
    failure but never adapts or grows.~~ **Done** (Phase 44): all four
    selectors (`Simple`, `Factor`, `IterationCount`, `StateChange`) plus
    `rampup_timesteps`, matching MATLAB bit for bit, and wired into
    `NonLinearSolver` as an adaptive ministep loop.
  - No `FlowStateBuilder` variants: fully implicit only, no explicit or
    adaptive-implicit (AIM) flux.

  *Smaller*
  - `upscale/` (4), `ad-core`'s own plotting (`plotWellSols`,
    `inspectFluidModel`), the packed-problem/restart machinery
    (`packSimulationProblem`, checkpointing, background runs), WENO upwinding,
    `WrapperModel` / composite models, and the state-function-graph
    *visualisation* tools (`plotStateFunctionGroupings`,
    `buildStateFunctionDigraph`) - the last of which is a real loss, since the
    graph is now the backbone and nothing draws it.
- **`co2lab`** - dissolution is rate-driven only, not the instantaneous
  `dis_rate = 0` variable-switching branch. (Not a gap: the `'S table'` VE
  relperm branch *is* ported, as `ve_fluids.make_ve_table_h`, with the typo
  that makes MRST's own version dead code fixed.)

## Whole modules not ported

Grouped by family. None of these have a `pymrst/modules` counterpart, with two
qualifications: `mpsaw` has a 439-line landing module whose lower subpackages
are stubs, and `hfm_v2`/`dfm`/`shale` exist only as re-export shims over `hfm`
(see the fourth band above). Neither counts as a port.

- **Alternative discretizations (TPFA is the only scheme):** `mpfa`, `re-mpfa`,
  `mpsaw`, `mimetic`, `nfvm`, `vag`, `dg`
- **Multiscale solvers:** `msrsb`, `msfvm`, `msmfem`, `domain-decomposition`.
  **Linear solvers are no longer the field-scale ceiling** (Phase 43):
  `ad_core.CPRSolver` gives the flow models AMG + ILU + GMRES with an iteration
  count that does not grow with the problem, and
  `ad_mechanics.PoroelasticSchurSolver` does the same job for coupled
  poroelasticity (a Schur complement on the constant mechanics block,
  preconditioned by fixed stress, 22x on Norne). What is left in
  `linearsolvers` is the multiscale machinery and the external bindings
  (`AMGCL`, `AGMG`), which are constant-factor wins, not new capability.
- **Grid generation / meshing:** `upr`, `gmsh`, `triangle`, `wellpaths`,
  `nwm`, `agglom`
- **Physics models:** `ad-eor` (polymer/surfactant), `solvent` (4-phase),
  `ad-scal`, `geochemistry`, `steady-state`
- **Geomechanics beyond vemmech/biot:** `fvbiot`, `fv-unsat`,
  `dual-continuum-mech`, and bit-for-bit MRST MPSA-W local Schur/tensor
  assembly. **`ad-mechanics` is done** for the
  poroelastic MRST surface (Phase 27):
  fully coupled poroelasticity single-phase *and* multiphase, fixed-stress
  splitting, the surface-uplift adjoint, and the Norne field case
  (`examples/run_norne_geomech.py`, which fetches the real field from OPM and
  runs it end to end). Phase 100 adds the von Mises plasticity material update;
  Phase 101 adds MPSA-W-style stress recovery utilities; Phase 102 and Phase
  104 add 2D and 3D global elastoplastic VEM Newton coupling; Phase 105 adds a
  node-centered MPSA-W interaction-region assembly surface with rigid/affine
  invariants.
- **Fractures beyond 2D EDFM:** exact MRST HFM mesh/workflow identity for
  `hfm_v2`, DFM, `hwu-fractures`, `shale`, and
  `dual-porosity-permeability` remains broader than the current compatibility
  surfaces and validated 2D/3D planar EDFM kernels.
- **`vemmech` model extras:** `LinearCompressibilityModel` is now present for
  the standalone water-only weakly-compressible TPFA branch. Its optional MRST
  thermal `linearTransport` branch is still unported.
- **CO2 / H2 extras:** `h2lab` (only `h2props` sampled), `hybrid-ve`,
  `co2lab-legacy`, `co2lab-demo`, `sleipner`
- **Workflow / infra / data:** `network-models` (GPSNet), `postprocessing`,
  `static-modeling`, `heterogeneity`, `spe10` (dataset), `test-suite`, `book`,
  `jutul` (Julia bridge), `libgeometry`, `mrst_api`

### Covered under a different name (not gaps)
- `adjoint` -> folded into `optimization` (flow) and `ad_mechanics.adjoint`
  (surface uplift, coupled poromechanics)
- `coarsegrid` -> folded into `upscaling`
- `co2lab-common` / `-ve` / `-spillpoint` / `-mit` -> single `co2lab` package

### Declared not portable
- `MICPMechModel` - `equationsMICPMech.m` is absent from the MRST tree (it
  lived in the deleted `ad-micp-mech` module).

## Layout differences from MRST

`README.md` says the tree "mirrors MRST directory-for-directory." That holds
*within* `pymrst/modules/` at the module level, but not at the top level, and it
glosses over two reshapings. The goal is for PyMRST to mirror MRST as closely as
Python allows; this section records where it does not yet.

### MRST layout - core dirs live at the repo root

```
mrst-2024a-Geomech/
  gridprocessing/  params/  plotting/  solvers/  utils/   <- core, at root
  modules/         <- every add-on: incomp, ad-core, co2lab-common, ...
  examples/        <- root examples, plus per-module examples inside each module
  output/  startup.m  Contents.m  README.txt  COPYING
```

### PyMRST layout - core dirs nested in the package

```
PyMRST/
  pymrst/                    <- the installable package (no MRST equivalent)
    gridprocessing/  params/  plotting/  solvers/  utils/
    modules/                 <- ported add-ons (incl. incomp)
  examples/  examples/validation/
  tests/  output/
  pyproject.toml  pytest.ini  README.md  GAPS.md  pymrst.egg-info/
```

### Where they diverge

| # | MRST | PyMRST | Mirror-able? |
|---|------|--------|--------------|
| 1 | core dirs at repo root | wrapped under `pymrst/` | No - Python needs a package root |
| 2 | `ad-core`, `co2lab-ve` (hyphens) | `ad_core`, `co2lab_ve` | No - hyphens are illegal in module names |
| 3 | `params/rock/*.m`, `params/wells_and_bc/*.m` (folders of files) | `params/rock/`, `params/wells_and_bc/` | Done - split into subpackages |
| 4 | `params/gravity.m` | `params/gravity/` plus `gravity_const.py` | Done - package wrapper delegates to the existing implementation |
| 5 | `co2lab-common` / `-ve` / `-spillpoint` / `-mit` (4 dirs) | `co2lab_common`, `co2lab_ve`, `co2lab_spillpoint`, `co2lab_mit` | Done - underscore packages mirror the MRST module family |
| 6 | `incomp/fluid/`, `incomp/transport/` | `incomp/fluid/`, `incomp/transport/` | Done - implementation files were reshaped into packages |
| 7 | `ad-props/props`, `simple`, `utils` | `ad_props/props`, `simple`, `utils` | Done for the implemented helper surface |
| 8 | `mpsaw/assembly`, `models`, `statefunctions`, `utils` | `mpsaw/assembly`, `models`, `statefunctions`, `utils` | Done for the implemented MPSA-W surface; `pymrst.modules.mpsaw` owns the code and old mechanics imports remain aliases |
| 9 | `vemmech/utils`, `grid_utils`, `plotting`, `ad_version`, `models` | matching underscore packages and landing files | Done for the implemented VEM/TEDFM/AD/calibration/plotting surface |
| 10 | examples scattered per module + a root `examples/` | one central `examples/` + `examples/validation/` | Partly - some module examples exist, but validation examples remain central |
| 11 | `Contents.m` index per directory | none | Cosmetic - Python package docstrings cover the important landing packages |

Items 1 and 2 are hard Python constraints and cannot be removed; the honest
target is "mirror MRST directory-for-directory *under* `pymrst/`, with hyphens
mapped to underscores." Items 3-6 are choices that can be undone to tighten the
mirror.

### To make PyMRST mirror MRST (proposed)

The remaining layout work is mostly breadth, not the core shape:

1. Add per-module `examples/` landing folders for modules that still rely only
   on the central `examples/validation/` tree.
2. Add more MRST subdirectory wrappers as additional APIs are ported
   (`ad-core/backends`, `linearsolvers`, `msrsb`, etc.).
3. Keep `MRST_LAYOUT_MAP.md` as the source of truth: a wrapper can exist only
   when it delegates to an implemented PyMRST symbol or explicitly documents a
   blocked/unported surface.

## Suggested priority

Ordered by leverage-to-effort, not by module size.

1. ~~**`verticalWell`** and the `pside`/`fluxside` BC helpers~~ - done. They
   were costing more than they looked: the boundary-face selection had been
   written out by hand, by coordinate comparison, in five separate modules.
   That is not just verbose, it is *wrong* on any grid that is not
   axis-aligned - `pside` uses the grid's direction tags.
2. ~~**`logNormLayers` / `gaussianField`**~~ - done. Their absence had been
   showing: every "heterogeneous" test case in the tree was a sum of sines,
   with no channels, no orders-of-magnitude contrast, and no tail - which is
   an especially poor look next to an ensemble history-matching module whose
   whole purpose is inferring exactly such fields.
3. ~~**`getFaceTransmissibility`**~~ - done. Three lines, inlined by hand in
   every model that needed a boundary condition.
4. ~~**The `ad-core` state-function graph**~~ - done (Phase 24). It was the
   highest-leverage item in the port: MRST's physics modules are *written as
   state functions*, so without the graph each of `ad-eor`, `solvent`,
   `ad-scal` and compositional flow is a bespoke residual to re-derive, and
   with it each is a close port. Validated by reproducing Phases 5a and 5b
   through it (oil-water bit-for-bit; SPE 1 to 1.6e-14).
5. **TEDFM fracture workflows** (3D EDFM + fracture mechanics beyond the
   validated VEM wrapper kernels) and the **CMG-export pipeline** -
   the README's stated next steps. `ROADMAP.md` plans everything else
   (Phases 25-42) in dependency order; the next gating items there are the
   core leftovers (Phase 25) and `upr` meshing (Phase 26).
6. ~~**Unstructured grids** (`triangleGrid`/`pebi`)~~ - done (Phase 23):
   `triangle_grid`, `tetrahedral_grid`, `pebi`, `make_layered_grid`, plus the
   grid-editing primitives (`add_faces`, `remove_faces`, `remove_cells`),
   all bit-exact against MATLAB. `upr` (the meshing pipeline on top of
   `pebi`) remains unported and is the natural follow-on.
7. **Compositional API parity is near its useful limit.** Phases 261-336 took
   `compositional` past MRST's file count, and the remaining items in that
   vein are aliases, catalogs and constructor overloads - real polish, but
   each one buys less than the last, and none of them buys physics. The
   leverage has moved: the highest-value work in the tree is now the Phase 25
   core leftovers (`processNNC`, `sliceGrid`, `grdecl2Rock`,
   `faceFlux2cellVelocity` and friends - cheap, mechanical, easy MATLAB
   dumps) because they gate the MPFA and fracture phases, and after those,
   `ad-eor`, which is the cleanest remaining test of the Phase-24
   state-function thesis on genuinely new physics.

## A caveat on the reference

Cross-validation turned up seven genuine defects in MRST 2024a, so "matches
MATLAB" is not always the same as "correct". They are documented in
`README.md`; in brief: `readAAIGrid` drops a data row at *each* end of the
file; `addVERelpermCapillaryFringe`'s `'S table'` branch raises on a
misspelled variable; `LimAzizShapeFactor` omits the permeability every
transfer function assumes it carries; `pollock.m` labels each streamline
point with the cell it is about to *enter*; `pebi` fails its own
`sortCellFaces` assertion on some valid flat Delaunay inputs;
`ad-mechanics` under-converges at its default tolerances, accumulating a 2e-7
drift across a schedule; and **`equationsOilWaterMech` omits the cell volume
on its Biot term**, weakening the poromechanical coupling by a factor of the
cell volume (a 10% error on 500 m3 cells, and dimensionally inconsistent, so
it depends on the grid's unit of length). The last is the most consequential
of the seven. Anyone treating the MATLAB tree as ground truth should know
about them.

---
Method: `.m` inventories from `../mrst-2024a-Geomech`, Python symbols from
`pymrst/**/*.py`; "missing" means grep of the Python tree returned no
counterpart. Depth notes come from reading the ports named above.

Last reconciled against the trees at **Phase 336**. The module table, the
"Core gaps" function lists, and the shim/stub classifications were regenerated
at that point; every function named missing under "Core gaps" was re-grepped
and none had appeared. The per-module depth narratives below that were not
re-verified line by line and may lag the tree - they are append-only phase
notes, and the `compositional` entry in particular has grown to ~790 lines,
which is a changelog rather than a gaps record.
