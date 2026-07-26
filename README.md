# PyMRST

A Python port of the [MATLAB Reservoir Simulation Toolbox](https://www.sintef.no/projectweb/mrst/)
(MRST 2024a), built alongside the reference MATLAB installation in
`../mrst-2024a-Geomech` and the two MRST textbooks (Lie 2019; Lie & Møyner 2021).

## Conventions

- **Strict SI units** everywhere; use `pymrst.units` constants to convert
  (`perm = 100 * units.MILLI * units.DARCY`).
- **0-based indexing** everywhere. Boundary faces are `-1` in
  `faces.neighbors` (MRST uses `0`).
- Data structures mirror MRST struct fields (`cells.facePos`,
  `faces.neighbors`) so the books and MATLAB code map directly onto the
  Python code.
- Every module is directly runnable: `python pymrst/utils/units.py` executes
  its unit tests.
- Every phase is validated against arrays exported from the MATLAB MRST
  installation (see `examples/validation/`).

## Install

```bash
pip install -e .              # numpy + scipy
pip install -e ".[plotting]"  # + pyvista (3D rendering)
pip install -e ".[gui]"       # + pyvistaqt (the interactive plot_toolbar)
```

## Testing

```bash
pytest                      # everything (173 tests, ~160 s)
pytest -m "not validation"  # module self-tests only
pytest -m validation        # replay the saved MATLAB cross-checks
```

## Run it on a real field

```bash
python examples/run_norne_geomech.py                # mini Norne, ~15 s
python examples/run_norne_geomech.py --subset full  # 7392 cells, ~2 min
python examples/run_norne_geomech.py --plot         # + the interactive inspector
```

Coupled flow and geomechanics on **Norne**, the Python equivalent of MRST's
`runNorneExample.m`. It downloads Norne's real corner-point grid and
petrophysics from OPM's `opm-tests` (the same pinned revision MRST uses,
~15 MB, once), builds the grid with `process_grdecl`, and injects water into
an oil-water reservoir fully coupled to poroelastic mechanics — then prints
the pressure, the saturation, the **surface uplift**, and four invariants that
would break if the coupling were wrong (see `report()` in that file). It needs
nothing but numpy and scipy; `--plot` adds a PyVista view.

Every module is also directly runnable and executes its own unit tests:
`python pymrst/utils/units.py`.

## Layout

The package tree mirrors MRST's directory-for-directory (hyphens become
underscores: `ad-core` -> `ad_core`), so the MRST books and MATLAB source
map straight onto the Python code:

| Package | MRST counterpart | Contents |
|---------|------------------|----------|
| `pymrst/gridprocessing` | `gridprocessing/` | grid constructors (Cartesian, corner-point, triangle/tet, PEBI, layered extrusion), grid editing, `computeGeometry` |
| `pymrst/params` | `params/` | rock, wells, BCs, gravity |
| `pymrst/solvers` | `solvers/` | `computeTrans`, `initResSol`/`initState` |
| `pymrst/plotting` | `plotting/`, `modules/mrst-gui/` | PyVista rendering: `plot_grid`, `plot_cell_data`, `plot_well`, streamlines, caprock/plume/trap maps, and `plot_toolbar` (interactive inspector) |
| `pymrst/utils` | `utils/` | units, ADI automatic differentiation |
| `pymrst/modules/incomp` | `modules/incomp/` | `incompTPFA`, incompressible fluids, explicit/implicit transport |
| `pymrst/modules/mpfa` | `modules/mpfa/` | MRST-layout legacy and general-corner tensor MPFA-O assembly; conservative single-phase solve with mixed BCs, sources, mobility, gravity, and rate/BHP wells |
| `pymrst/modules/nfvm` | `modules/nfvm/` | corrected HAPs, 2-D/3-D NTPFA with fully implicit AD weights, AvgMPFA, Hybrid, and AD-core installers |
| `pymrst/modules/fv_unsat` | `modules/fv-unsat/` | hydraulic soil catalog, van Genuchten-Mualem laws, and a conservative fully implicit Richards/MPFA time-step model |
| `pymrst/modules/re_mpfa` | `modules/re-mpfa/` | direct ports of hydraulic-head arithmetic/upstream MPFA weighting, Newton-AD iteration, and adaptive time stepping |
| `pymrst/modules/mimetic` | `modules/mimetic/` | MRST-layout mimetic inner-product families and conservative hybrid incompressible pressure/flux solve with gravity and rate/BHP wells |
| `pymrst/modules/ad_core` | `modules/ad-core/` | the state-function graph (`StateFunction`, groupings, PVT/flow/flux evaluators), the `PhysicalModel`/`ReservoirModel` protocol, the facility (wells), `NonLinearSolver`, TPFA operators, schedule driver |
| `pymrst/modules/ad_props` | `modules/ad-props/` | `initSimpleADIFluid`, deck fluids, PVT interpolation |
| `pymrst/modules/ad_blackoil` | `modules/ad-blackoil/` | oil-water and three-phase black-oil models |
| `pymrst/modules/deckformat` | `modules/deckformat/` | ECLIPSE deck input (SPE 1 subset, GRDECL) |
| `pymrst/modules/vemmech` | `modules/vemmech/` | 2D/3D VEM linear elasticity, Biot poroelasticity, strain/stress operators |
| `pymrst/modules/ad_mechanics` | `modules/ad-mechanics/` | coupled flow + geomechanics: single-phase and multiphase fully-coupled poroelasticity, fixed-stress splitting, surface-uplift adjoint gradients, the Mandel benchmark |
| `pymrst/modules/ad_micp` | `modules/ad-micp/` | MICP reactive transport |
| `pymrst/modules/hfm` | `modules/hfm/` | embedded discrete fractures (EDFM) |
| `pymrst/modules/hfm_v2` | `modules/hfm_v2/` | MRST-shaped HFM v2 compatibility surface over the validated HFM kernels |
| `pymrst/modules/dfm` | `modules/dfm/` | DFM compatibility helpers over the validated EDFM/projection kernels |
| `pymrst/modules/shale` | `modules/shale/` | shale/HFM compatibility helpers over the validated EDFM/projection kernels |
| `pymrst/modules/co2lab` | `modules/co2lab-common`, `-ve`, `-spillpoint`, `-mit` | CO2/H2 properties (sampled tables **and** correlations), brine-gas model, vertical equilibrium with dissolution, trap analysis, trapping inventory, injection-point ranking, the NPD Storage Atlas |
| `pymrst/modules/diagnostics` | `modules/diagnostics/` | time-of-flight, tracers, sweep |
| `pymrst/modules/upscaling` | `modules/coarsegrid/`, `modules/upscaling/` | grid coarsening, flow-based permeability upscaling |
| `pymrst/modules/sequential` | `modules/sequential/` | sequential pressure/transport splitting |
| `pymrst/modules/compositional` | `modules/compositional/` | Peng-Robinson EOS, flash, phase density/viscosity, flash-to-flow-props, differentiated flash, overall-composition flow model |
| `pymrst/modules/streamlines` | `modules/streamlines/` | Pollock semi-analytic streamline tracing |
| `pymrst/modules/optimization` | `modules/optimization/` | adjoint gradients, NPV rate optimization |
| `pymrst/modules/dual_porosity` | `modules/dual-porosity/` | shape factors, matrix-fracture transfer, DP flow models |
| `pymrst/modules/geothermal` | `modules/geothermal/` | Spivey/Kestin water-brine EOS, coupled mass-energy |
| `pymrst/modules/ensemble` | `modules/ensemble/` | Monte Carlo ensembles, EnKF/ES-MDA history matching |

## Roadmap

Phases 0-23 below are what has been **built**. `GAPS.md` records what has not,
and `ROADMAP.md` plans the remaining phases (24-42) - the state-function graph,
geomechanics, the discretization shelf, EOR, multiscale solvers - in dependency
order, with the validation each one needs.

1. ~~Phase 0 — package scaffolding, units system~~
2. ~~Phase 1 — grid kernel: `cart_grid`, `tensor_grid`, `compute_geometry`
   (validated field-by-field against MATLAB MRST 2024a)~~
3. ~~Phase 2 — rock/wells/BC, `compute_trans`, incompressible TPFA solver
   (validated against MATLAB: quarter five-spot, mixed BCs, wells, gravity)~~
4. ~~Phase 3 — two-phase incompressible transport: Corey fluids,
   explicit/implicit upwind solvers, sequential splitting (validated
   against the analytic Buckley–Leverett solution and MATLAB)~~
5. ~~Phase 4 — ADI automatic differentiation, TPFA operators, compressible
   single-phase AD simulator (matches MATLAB to machine precision)~~
6. ~~Phase 5a — AD-OO framework (`ad_core` lite): fully-implicit two-phase
   oil-water model with MRST's standard well model, schedule driver
   (matches MATLAB ad-blackoil to ~1e-14 relative)~~
   ~~Phase 5b — three-phase black-oil with dissolved gas (variable
   switching), minimal SPE 1 deck reader, live-oil PVT tables; SPE 1
   matches MATLAB ad-blackoil to ~1e-10 relative~~
7. ~~Phase 6 — corner-point grids: `process_grdecl` with fault-face
   subdivision, ACTNUM, and pinch-outs; GRDECL reader (SPE1, faulted
   synthetic, and the full Johansen NPD5 field match MATLAB bit-exactly)~~
8. Phase 7 — research modules:
   ~~7a geomechanics: vemmech VEM linear elasticity, including the 2D
   plane-strain operator path (`validate_phase61.py` matches MATLAB's
   `calculateQF`, `VEM2D_div`, `Enu2C`, and `VEM_assemble`), VEM stress
   utilities (`validate_phase68.py` matches `calculateStressVEM` and
   `calStressEigsVEM`), TEDFM VEM wrapper kernels
   (`validate_phase69.py` matches `calculateQFTEDFM`, `calculateQCTEDFM`,
   `C2DTEDFM`, `VEM3D_divTEDFM`, `VEM_divTEDFM`, and
   `VEM_assembleTEDFM` on `G.Matrix` wrapper grids; `validate_phase70.py`
   matches `VEM_linElastTEDFM` displacements, stress/strain, reduced system,
   and `disc` operators; `validate_phase71.py` matches `flipGrid`,
   `padGrdecl`, and `verticalGrdecl`; `validate_phase72.py` matches the
   deformed plotting geometry helpers; `validate_phase73.py` matches the
   first AD VEM tensor utilities; `validate_phase78.py` matches the numeric
   `VEM_assemble_AD`/`VEM_linElast_AD` path; `validate_phase79.py` checks
   3D material AD `Ax_derivs` against central finite differences;
   `validate_phase81.py` checks the corresponding 2D material AD path;
   `validate_phase80.py` checks the calibration setup's eliminated objective;
   `validate_phase82.py` checks the reduced VEM residual used by the
   adjoint-structured calibration gradient;
   `validate_phase83.py`/`validate_phase84.py` check AD-aware material
   closures and their use in stiffness-control adjoint columns;
   `validate_phase85.py` checks boundary-condition derivative columns;
   `validate_phase86.py` checks load/density derivative columns;
   `validate_phase87.py` checks the calibration objective displacement
   partial against free-DOF finite differences;
   `validate_phase88.py` checks the calibration objective control partial
   against fixed-displacement control finite differences;
   `validate_phase89.py` checks analytic local stress-observation objective
   derivatives against stress-space finite differences;
   `validate_phase90.py` checks the optional eliminated-objective ADI2
   Hessian output and Hessian-direction callback;
   `validate_phase91.py` checks the calibration workflow driver, synthetic
   data mode, optimization bookkeeping, and objective reduction;
   `validate_phase92.py` checks the MRST-style SR1 trust-region optimizer
   slice used by the calibration workflow;
   `validate_phase93.py` checks SR1 backup/resume files and progress hooks;
   `validate_phase94.py` checks first-order ADI coefficient propagation
   through the eager VEM sparse-tensor substrate;
   `validate_phase95.py` checks the pure-Python lazy tensor expression graph
   and exact second-order AD Hessian substrate;
   `validate_phase96.py` checks eliminated-objective Hessians when K/G/rho
   material controls are active;
   `validate_phase97.py` checks MATLAB-style deformed plotting handles;
   `validate_phase98.py` checks VEM mechanics invariants on triangle, PEBI
   polygonal, and tetrahedral grids;
   `validate_phase99.py` checks EDFM-to-TEDFM mechanics workflow coupling and
   fracture normal-strain/aperture-change recovery;
   `validate_phase100.py` checks the von Mises plasticity material update;
   `validate_phase101.py` checks MPSA-W-style stress recovery utilities;
   `validate_phase102.py` checks the global elastoplastic VEM Newton coupling;
   `validate_phase103.py` checks 3D planar EDFM projection/flow and
   mechanics-driven 2D fracture propagation;
   `validate_phase104.py` checks 3D elastoplastic VEM coupling;
   `validate_phase105.py` checks MPSA-W interaction-region assembly
   invariants;
   `validate_phase106.py` checks the hfm_v2/DFM/shale compatibility surfaces
   and toughness-driven propagation;
   `validate_phase107.py` checks 3D planar EDFM physical parity against a
   MATLAB MRST HFM dump;
   `validate_phase108.py` checks the standalone vemmech
   `LinearCompressibilityModel` water-only linear TPFA branch;
   `validate_phase74.py` matches the first
   calibration helper slice; `validate_phase75.py` matches the calibration
   boundary-condition helper; `validate_phase76.py` matches the numeric
   calibration stress helper; `validate_phase77.py` matches the numeric
   Gaussian objective helper), and the
   water, oil-water, and black-oil fully coupled/fixed-stress branches of
   `run2DCase.m`
   (`validate_phase63.py`, `validate_phase64.py`, `validate_phase62.py`,
   `validate_phase65.py`, `validate_phase66.py`, `validate_phase67.py`) +
   fully coupled linear Biot poroelasticity
   (matches Terzaghi's analytic consolidation to 0.2%)~~;
   ~~7b MICP: reactive transport (microbes, oxygen, urea, biofilm,
   calcite) with clogging permeability. Verified by evaluating MRST's own
   equationsMICP residual at the PyMRST states: 7.6e-15 (see
   `examples/validation/check_phase7b_residual.m`). Note MRST's own solver
   leaves a ~1e-8 residual in its microbe equation~~;
   ~~7c EDFM: 2D embedded discrete fractures (conductivity index with an
   exact average-distance integral, matrix-fracture and fracture-fracture
   NNCs, coupled TPFA solve) - matches MATLAB hfm to machine precision~~;
   HFM v2/DFM/shale compatibility landing modules, 3D planar EDFM projection
   validation against MRST physical reference quantities, and toughness-driven
   fracture propagation are present. Remaining fracture work is exact
   MRST mesh/workflow identity for the larger HFM/shale examples and richer
   propagation physics beyond the focused validated kernels.
   MICPMechModel is not portable: `equationsMICPMech.m` is missing from
   the MRST tree (it lived in the deleted `ad-micp-mech` module).
9. ~~Phase 8 — CO2/H2 storage (`co2lab`): sampled CO2 and H2 property
   tables (AD-transparent bilinear interpolation) and a fully-implicit,
   buoyancy-driven brine-gas model. CO2 injection matches MATLAB's
   `TwoPhaseWaterGasModel` to 1e-13 (bhp to 4e-4, see validate_phase8.py).~~

10. ~~Phase 9 - vertical equilibrium (`co2lab-ve`): `top_surface_grid`
    collapses a 3D columnar aquifer onto its caprock, `average_rock`
    upscales the rock, and the sharp-interface VE fluid gives upscaled
    relperms and capillary pressure. Matches MATLAB co2lab-ve to machine
    precision. Long-term CO2 migration in 2D instead of 3D.~~

11. ~~Phase 10 - structural trap analysis (`co2lab-spillpoint`): a priority
    flood of the caprock finds every trap, its spill depth, catchment,
    trapping capacity, and the spill graph linking traps. Matches MATLAB's
    cell-based `trapAnalysis` cell-for-cell and to machine precision.~~

12. ~~Visualization (`pymrst/plotting`): PyVista rendering of any PyMRST
    grid - Cartesian, corner-point with faults, or a VE top surface - as
    VTK polyhedral cells, plus `plot_well` and CO2-specific caprock, plume,
    and trap maps. See `examples/co2_storage_workflow.py` for the whole CO2
    chain end to end.~~

13. ~~Flow diagnostics (`diagnostics`): time-of-flight (forward and
    backward), well tracer partitions, F-Phi curves, Lorenz coefficient and
    sweep efficiency - the standard tools for ranking well placements from
    a single pressure solve. Matches MATLAB to machine precision.~~

14. ~~Coarsening and upscaling (`upscaling`): partition a fine grid, build
    the coarse grid, and upscale permeability with sealed-sides flow-based
    upscaling. Matches MATLAB's `upscalePerm` to 9e-16.~~

15. ~~Sequential splitting (`sequential`): pressure and transport solved in
    turn rather than fully implicitly, wrapping the validated AD oil-water
    model. Outer iterations contract to the fully implicit solution.~~

16. ~~Compositional (`compositional`): Peng-Robinson EOS, Wilson K-values,
    Rachford-Rice, and a two-phase flash with a trivial-solution guard.
    Matches MATLAB's `standaloneFlash` to 1e-12.~~

17. ~~Streamlines (`streamlines`): Pollock's semi-analytic tracing, forward
    and reverse, with time-of-flight along each line. Matches MATLAB's
    `pollock` to 6e-14 in coordinates and 1e-16 in time-of-flight.~~

18. ~~Adjoint gradients and optimization (`optimization`): the gradient of
    an NPV objective with respect to every well control of every timestep,
    from one backward sweep instead of one simulation per control, plus
    projected gradient ascent on the rates.~~

19. ~~Interactive inspector (`plot_toolbar`, MRST's `mrst-gui/plotToolbar`):
    scrub a whole schedule, switch datasets, transform (log10, abs),
    threshold, drag a clip plane, and read a histogram of what is shown.
    Uses `pyvistaqt` for a real non-blocking window with menus when it is
    installed (`pip install 'pymrst[gui]'`), and degrades to a plain
    key-driven PyVista window when it is not. See `examples/toolbar_demo.py`.~~

20. ~~Dual porosity (`dual_porosity`): Kazemi / Warren-Root / Coats /
    Lim-Aziz shape factors, single- and two-phase matrix-fracture transfer
    with upstream-weighted relperm, and the WaterModelDP and
    TwoPhaseOilWaterModelDP flow models. Matches MATLAB to 3e-16
    (single-phase) and 1e-9 (two-phase).~~

21. ~~Geothermal (`geothermal`): the Spivey et al. (2004) water and brine
    density and Kestin viscosity correlations, AD-transparent and matching
    MATLAB to 2e-16 over 432 states; plus a fully-implicit mass-and-energy
    model with advective and conductive heat transport. Verified against
    closed forms: steady conduction is linear to 1e-3 K, a sealed box
    conserves energy to 1e-16, and a thermal front is retarded by the
    predicted (rock + fluid) / fluid heat-capacity ratio to 2%.~~

22. ~~Ensembles (`ensemble`): Monte Carlo over uncertain parameters, and
    EnKF / ES-MDA history matching. The analysis step recovers the exact
    Kalman posterior (mean *and* covariance) on a linear-Gaussian problem,
    and ES-MDA recovers zone permeabilities from well data end to end.~~

23. ~~CO2 trapping inventory (`co2lab.inventory`): the headline output of a
    storage assessment - the injected CO2 split into dissolved, residually
    trapped, structurally trapped, sub-scale trapped and free/mobile, over
    time. Matches MATLAB's `massTrappingDistributionVEADI` to 1.5e-15 in all
    seven categories. Plus `maximize_trapping` and
    `find_optimal_injection_point`, which rank injection points by the spill
    tree they reach rather than by the trap they sit in (trees and capacities
    match MATLAB to 2e-14).~~

24. ~~CO2-brine correlations (`co2lab.co2_brine_props`, MRST's `co2lab-mit`):
    Redlich-Kwong CO2 density with Maxwell root selection, Fenghour CO2
    viscosity, Rowe-Chou brine density, Islam-Carlson brine+CO2 viscosity,
    Duan-Sun activity (the salting-out coefficient a dissolution model needs)
    and the Davidson gas-mixture rule. A computed alternative to the sampled
    tables in `co2props`. Matches MATLAB to 1e-15 over 540 states.~~

25. ~~Dissolution in the VE model (`co2lab.ve_dissolution`): CO2 going into
    solution as a third conserved quantity, carried by the brine and fed by
    the free plume at a rate limited by the interface area, by how much more
    the brine can hold, and by how much free CO2 there is. The slowest
    trapping mechanism and the safest - dissolved CO2 sinks rather than rises
    and can never leak - and it is what finally puts something in the
    `dissolved` bar of the trapping inventory.~~

26. ~~The Norwegian CO2 Storage Atlas (`co2lab.atlas`): the NPD's mapped
    formations, read from their ESRI ASCII rasters and turned into
    corner-point grids that feed the ordinary
    `process_grdecl -> top_surface_grid -> trap_analysis` chain. Matches
    MATLAB's `getAtlasGrid` bit for bit: the rasters exactly, ACTNUM cell for
    cell (79,422 active cells for Johansen), ZCORN and COORD to 5e-13 m. Real
    Johansen has 730 structural traps holding 5.25 km3 of pore space.
    Needs the `CO2Atlas` dataset (a 16 MB download from SINTEF, unpacked into
    MRST's data directory); point `MRST_DATA_DIR` at that directory.~~

27. ~~Upscaled VE fluids (`co2lab.ve_fluids`): the capillary fringe the
    sharp-interface model pretends away. `free_sg` (the mobile part of the
    CO2, once a retreating plume has stranded the rest), the analytic
    cap-linear model, and the general capillary-fringe model, which
    *integrates* an arbitrary fine-scale pc and kr curve vertically into
    upscaled tables. Matches MATLAB's `addVERelpermCapillaryFringe`
    ('P-scaled table') and `free_sg` to 4e-16.~~

28. ~~Node-based trap analysis (`co2lab.node_trapping`): traps resolved on
    the grid's *nodes* rather than its cell centres, so a trap's rim can fall
    between cells. A different algorithm, not a variant of the cell-based one:
    a spill field over node adjacency **including diagonals**, then repeated
    **cycle contraction** - two regions that spill into each other fill to a
    common level and are one trap, so cycles in the spill graph are collapsed
    and the spill points recomputed until none remain. Matches MATLAB's
    `computeNodeTraps` and `trapAnalysis(Gt, false)` exactly: node traps
    node-for-node, cell traps cell-for-cell, spill depths to 0.~~

29. ~~Core helpers (`params`, `solvers`, `utils`): the things every MRST
    script assumes and PyMRST had been doing by hand - `pside` / `fluxside`
    (a boundary condition on a *side*, found by the grid's direction tags
    rather than by testing coordinates against a tolerance), `vertical_well`
    (a well down a logical (i, j) column, skipping inactive cells),
    `get_face_transmissibility` (the harmonic reduction of the half-face
    values), and `gaussian_field` / `log_norm_layers` (real layered
    log-normal heterogeneity instead of a sum of sines). All match MATLAB
    exactly; the two field generators are checked by replaying MATLAB's own
    random draw through the Python smoother, so what is compared is the
    algorithm and not the RNG.~~

30. ~~Unstructured grids (`gridprocessing`, Phase 23): `triangle_grid` and
    `tetrahedral_grid` (a valid grid from any point set and simplex list,
    Delaunay by default), `pebi` (the Voronoi dual of a triangular grid -
    perpendicular-bisector faces, with MRST's auxiliary-triangle repair
    when a boundary triangle's circumcenter falls outside the domain),
    `make_layered_grid` (extrude any 2D grid to layered 3D), and the
    grid-editing primitives they stand on (`add_faces`, `remove_faces`,
    `remove_cells`, `empty_grid`). All five validation cases - triangle,
    two pebi (with and without the auxiliary path), tet, layered - match
    MATLAB bit-for-bit in topology and node coordinates and to machine
    epsilon in volumes and centroids (`validate_phase23.py`). The
    triangulations are dumped from MATLAB, so the comparison isolates the
    grid construction from the triangulator.~~

31. ~~The state-function graph (`ad_core`, Phase 24): the machinery MRST's
    AD-OO framework is actually built on, and the thing this port had been
    doing without. A model is no longer one function that assembles a
    residual top to bottom; it is a *graph* of small property evaluators -
    `StateFunction`, `StateFunctionGrouping` - each declaring what it
    depends on, which the solver walks, evaluating every property at most
    once per state and caching it. Three groupings (`PVTPropertyFunctions`,
    `FlowPropertyFunctions`, `FlowDiscretization`), the
    `PhysicalModel`/`ReservoirModel` protocol, a `FacilityModel` carrying
    one well model instead of one per fluid, and a real `NonLinearSolver`
    (relaxation, line search, timestep cutting, convergence reports) in
    place of the fixed Newton loop.

    **Why it matters more than its lack of new physics suggests:** MRST's
    physics modules - `ad-eor`, `solvent`, `ad-scal`, compositional flow -
    are *written as state functions*. Without the graph each is a bespoke
    residual to re-derive; with it each is a close port. This is the phase
    the rest of the roadmap stands on.

    **How it is validated:** the graph has no MATLAB array to dump - it is
    machinery, not physics - so it is held to a stronger standard instead.
    `GenericBlackOilModel` assembles both validated cases entirely from the
    graph and must reproduce them: oil-water matches the hand-rolled model
    **bit for bit** (0.0), and SPE 1 - three-phase, live oil, variable
    switching and flash - to 1.6e-14. Both still match MATLAB at the
    tolerances Phases 5a and 5b matched at (2e-14 and 2.4e-11 relative).
    See `examples/validation/validate_phase24.py`. The hand-rolled models
    are kept: they are what the graph is checked against.~~

32. ~~Coupled flow and geomechanics (`ad_mechanics`, Phase 27): the module
    this reference tree is named after. `MechanicModel` (poroelastic
    mechanics on the VEM discretization), `MechWaterModel` (fully coupled),
    and `FixedStressSplitModel` (the sequential scheme that makes coupled
    poromechanics affordable at field scale). `vemmech` gained the strain
    and stress operators and `invCi` that the coupling needs.

    **The graph paid off exactly as Phase 24 predicted.** All of
    poroelasticity, as far as the flow equation is concerned, is one
    property - the pore volume grows with the rock's dilation,
    `V(phi + alpha div(u))`. So the coupling *is* a `PoreVolume` state
    function, and the two models differ only in which one they install
    (`PoroMechanicalPoreVolume` vs `FixedStressPoreVolume`). Nothing
    downstream - component mass, the conservation equations - knows that
    mechanics happened.

    **Validated three ways, and the third is the one that counts:**
    against MATLAB `ad-mechanics` (4e-9 pressure, 9e-10 displacement);
    against Terzaghi's analytic consolidation (0.3%); and against
    **Mandel**, where the pressure at the centre of a squeezed, side-drained
    slab *rises above* its initial value before decaying - the Mandel-Cryer
    effect. A one-way-coupled, lagged, or mis-scaled model decays
    monotonically and sails through Terzaghi regardless; only correct
    coupling turns the pressure back up. PyMRST reproduces the overshoot
    (peak 1.096 vs analytic 1.098) and the whole profile to under 1%, and
    the error falls under refinement. Fixed-stress splitting contracts to
    the fully coupled answer to 5e-7.~~

33. ~~Multiphase poroelasticity (`ad_mechanics.MechBlackOilModel`, MRST's
    `MechOilWaterModel` / `MechBlackOilModel`): two- or three-phase
    black-oil, with wells, coupled to mechanics. It is `GenericBlackOilModel`
    with `PoroMechanicalPoreVolume` installed and the displacement added as a
    primary variable - about 150 lines, because the multiphase physics needed
    no re-derivation whatsoever. The third and cheapest dividend from Phase
    24. Validated against MATLAB (see the seventh finding, immediately
    below).~~

34. ~~The geomechanics adjoint (`ad_mechanics.adjoint`, MRST's
    `runAdjointExample` / `objUplift`): the gradient of **surface uplift** -
    the vertical displacement of a chosen surface node, integrated over the
    schedule - with respect to every well control at every timestep, from one
    backward sweep instead of one simulation per control. This is the question
    coupled geomechanics is actually asked in practice: *how should the wells
    be operated to keep the ground still?*

    Verified as the flow adjoint was (Phase 18), by finite differences rather
    than against MATLAB - a gradient is either the derivative of the forward
    model or it is not, and the forward model here is the coupled poroelastic
    one already validated against MATLAB, Terzaghi and Mandel, so the chain is
    closed. The full uplift gradient matches central differences to **2.8e-10**
    relative; the two blocks the adjoint adds are finite-differenced
    separately as well (`dR/du` exactly, `dR/dx_{n-1}` to 4.5e-9), since one
    end-to-end check can hide compensating errors between them. The sign is
    pinned by its own test - injection lifts the ground, and a flipped sign
    still passes a gradient check.

    Porting it exposed a latent trap in the Phase-24 graph:
    `component_conservation_equations` wrapped the previous state's mass in
    `value()`. Harmless in the forward Newton (where the previous state is
    plain numbers anyway), but it silently zeroed the entire `dR/dx_{n-1}`
    block the moment the adjoint seeded that state as an AD variable. The
    hand-rolled models carried a comment warning about exactly this; the graph
    had lost it. It is back, and `test_jac_wrt_previous_matches_finite_
    difference` now pins it.~~

35. ~~The Norne field case (`examples/run_norne_geomech.py`, MRST's
    `runNorneExample`): coupled flow and geomechanics on the **real field**,
    runnable end to end. `deckformat.norne` fetches Norne's corner-point grid
    and petrophysics from OPM's `opm-tests` at the revision MRST pins, and
    assembles them as `makeNorneGRDECL.m` does - including the `COPY
    PERMX -> PERMY, PERMZ` and the 22 per-layer `PERMZ` multipliers, one of
    which is exactly zero (a sealing shale). `gridprocessing.cut_grdecl`
    (MRST's `cutGrdecl`) takes the sub-box out of the raw arrays before the
    grid is built.

    The example prints four invariants, and they are the point: the field
    starts in **exact equilibrium** (a confining pressure of pRef cancels a
    pore pressure of pRef at alpha = 1, so |u| = 3e-15 m before injection),
    the pressurised reservoir dilates, injection **lifts the surface**, and
    the water balance closes to 5e-13 *through the poromechanical pore
    volume* - which it only can if the mechanics and the flow agree about how
    far the rock opened.

    **And it is validated against MRST on the field itself**
    (`validate_norne.py` / `dump_norne_reference.m`): running the identical
    case through MRST's `MechOilWaterModel`, saturations agree to **7e-9**,
    full displacements to **2.2e-7**, vertical displacements to **1.6e-7**,
    volumetric strain to **3.2e-7**, full strain to **4.4e-7**, effective
    stress to **4.4e-7**, total stress to **4.3e-7**, bhp to **6.7e-8**,
    pressures to **4.6e-7**, and both codes start in equilibrium
    (|u| = 3e-15 m).

    That comparison also puts a number on the seventh finding, on a real field
    rather than a test box: MRST's missing cell volume **over-pressures Norne
    by a factor of 16.6** (8443 bar against 509) and predicts **46.9 m of
    surface uplift** against 1.6 m. Norne's cells hold ~70,000 m3, so the
    mechanical storage MRST drops is not ~700 m3 per cell but ~0.01 m3: the
    rock stops absorbing the injected water, and with a nearly incompressible
    fluid the pressure has nowhere to go. On a unit-cell grid the defect is
    invisible. On Norne it is the whole answer.~~

36. ~~Well connection pressure drops (`FacilityModel.connection_pressure_drop`,
    MRST's `updateConnectionPressureDrop`). A well reports its pressure at a
    reference depth but feels the reservoir at its perforations, and the fluid
    column between them weighs something: on Norne the wells sit 32 m and 62 m
    below their datum, worth 3 and 6 bar. PyMRST's well model had `cdp = 0`
    hardcoded - exact for SPE 1 (where `refDepth` *is* the perforation depth,
    which is why Phases 5b and 24 never noticed) and wrong everywhere else.
    Found by the Norne comparison, which is what a real-field cross-check is
    for.

    Two details of MRST's that matter and are easy to miss: the wellbore
    density is **frozen at the first Newton iteration** of each step (making it
    implicit would couple every perforation of a well to every other and fill
    in the Jacobian), and it is built from the well's *connection fluxes*, not
    from its `compi`.~~

37. ~~A linear solver for coupled poroelasticity
    (`ad_mechanics.PoroelasticSchurSolver`). Profiling the Norne run showed the
    cost is not where anyone would guess: assembling the residual takes
    **0.06 s**, and the sparse *direct solve* of the resulting 45,000 x 45,000
    system takes **34.9 s per Newton iteration**. The run was entirely a
    linear-algebra problem.

    Two facts fix it. **The mechanics block never changes** - linear elasticity,
    so the stiffness matrix is identical at every Newton iteration of every
    timestep. Factorise it once (7.4 s) and eliminate the displacement against
    the cached factorisation, applying the Schur complement as an operator
    rather than forming it. That alone is 8x. **And fixed stress is the right
    preconditioner:** the eliminated block is, cell by cell, almost exactly the
    fixed-stress storage `(b s / dt) alpha^2 V / K_dr` - the same `1/K_dr` the
    fixed-stress *splitting* uses, for the same reason. Adding it to the flow
    block's diagonal cuts GMRES from 122 iterations to 36.

    **34.9 s -> 1.6 s per Newton iteration (22x).** On identical work: two
    Norne timesteps go from 361.7 s to 24.4 s (14.8x) and the states agree to
    **2e-15** in pressure, saturation, displacement and bhp. The full 21-step
    schedule now runs in **2 minutes** instead of about an hour. It is the
    direct solution, reached differently, and
    `test_reproduces_the_direct_solution` pins that rather than trusting it.

    `NonLinearSolver` now takes a `linear_solver`, so this is the first
    pluggable solver in the port; `--direct` in the Norne example restores the
    direct solve it is checked against.~~

38. ~~Capillary pressure (`ad_core`, Phase 38). Until now **every model in this
    port assumed `p_w = p_o = p_g`** - there was no `Pc` anywhere in the graph,
    and `PressureGradient` returned a single gradient that all phases shared.
    So water could only move where the *oil* pressure moved it, and capillary
    imbibition was not merely inaccurate, it was identically zero.

    The chain, mirroring MRST: `CapillaryPressure` (water carries `-pcOW`) ->
    `PhasePressures` (`p_alpha = p + pc_alpha`) -> `PressureGradient`, which now
    returns **one gradient per phase** -> `PhasePotentialDifference` ->
    `PhaseFlux`.

    Validated against MRST's own state-function model (`GenericBlackOilModel`)
    on the same case run **twice**, with and without capillarity: the no-pc case
    still matches to **1e-12** (so the change is inert where it should be, and
    Phases 5/5b/24/27 are undisturbed), the pc case matches to **1.3e-12**, and
    the two differ by 0.015 saturation units - so the test is not passing on a
    capillary pressure that does nothing.

    **One subtlety worth the trouble it caused.** The pc case first matched only
    to 1.3e-6 while the no-pc case matched to 1e-12 - an asymmetry that had to
    mean something in the pc path was still wrong. It was: MRST's
    `ShrinkageFactors` and `Viscosity` both set `usePhasePressures = true`, so
    `b_w` and `mu_w` are evaluated at the *water* pressure `p - pcOW`, not at
    the model's reference pressure. A small effect, invisible until a capillary
    pressure is switched on, and worth 6 orders of magnitude of agreement.~~

39. ~~Well control switching (`ad_core`, Phase 39). A real schedule is never
    "inject at 1 l/s" - it is "inject at 1 l/s *unless* that needs more than
    300 bar, in which case hold 300 bar". PyMRST's wells had **no limits at
    all**: a rate-controlled injector would demand whatever bottom-hole
    pressure its rate required, however unphysical.

    Wells now carry `lims` (MRST's `W.lims`), and
    `FacilityModel.update_switched_controls` (MRST's
    `updateSwitchedWellControls`) switches a well to the limit it breaks, in
    the middle of the Newton loop - which means the step cannot be declared
    converged, because it is now solving a different problem. All seven control
    equations a well can switch *into* are implemented (`bhp`, `rate`, `orat`,
    `wrat`, `grat`, `lrat`, `vrat`), since a well that hits its liquid-rate
    limit otherwise has nowhere to go.

    Validated against MRST twice: once with limits that bite (both wells
    switch, and **the control mode of every well at every step** matches, along
    with the pressures and rates that follow - to 2e-12), and once with the
    limits placed out of reach, where nothing may switch. That second case is
    the one that matters most: it checks the machinery is inert when it should
    be, which is the failure mode that would quietly corrupt every previously
    validated phase.~~

40. ~~The component framework (`ad_core`, Phase 40) - the gap that was gating
    the rest of the roadmap. The conservation equations hardcoded "one
    component per phase", which is true for black-oil and false for nearly
    everything else: polymer and surfactant are species dissolved *in* a phase,
    a compositional species lives in two phases at once, a tracer lives in one
    but is not it.

    Now a **component** is an object that says only *how much of itself is in
    each phase* (`component_density`), and the equations loop over
    `model.components`:

        mass_i = sum_ph  pv * s_ph * density[i][ph]
        flux_i = sum_ph  upwind(flag_ph, density[i][ph]) * v_ph

    Dissolved gas stops being a special case and becomes a `GasComponent`
    whose density is simply non-`None` in two phases. Black-oil is reproduced
    to machine precision (8.5e-16 oil-water, 1.6e-14 SPE 1) and **all six
    MATLAB validations still pass** - the refactor touched the core of every
    model in the port.

    **The proof it works is a tracer** (`ad_blackoil/models/tracer_black_oil_model.py`):
    `TracerBlackOilModel` is `GenericBlackOilModel` plus *one entry in
    `self.components`* and one primary variable. The conservation equation, the
    flux, the upwinding, the wells, the Newton solver and the linear solve are
    untouched - they never knew how many components there were. And the test
    that matters is exact rather than statistical: a passive tracer must not
    perturb the water it rides in **at all**, so changing the tracer field must
    leave the flow residual *bit-identical*. It does. (It is also conserved to
    5.7e-11 and never gets ahead of the water.)

    A passive tracer is polymer with the interesting parts removed. `ad-eor`
    adds the viscosity multiplier and the adsorption - it does not add any of
    the plumbing.~~

41. ~~Boundary conditions and sources for the AD models (`ad_core`, Phase 41).
    The AD models could be driven by **wells and nothing else**. A pressure
    boundary, a flux boundary, an aquifer standing in for the rest of the
    field, a source term - none of them existed, so every case in the port had
    to be a closed box with a well in it. That is not how a reservoir sits in
    the world, and it is not how most of MRST's own examples are set up.

    `boundary_conditions.py` ports `addFluxesFromSourcesAndBC` /
    `getBoundaryConditionFluxesAD` / `getSourceFluxesAD`. Which mobility
    applies at a pressure boundary depends on which way the fluid is going:
    fluid *leaving* carries the cell's own mobility, fluid *entering* carries
    the total mobility split by the boundary's saturation - you cannot use a
    rock's relative permeability to a fluid that is not in it yet.

    **Better than MRST here, for once.** MRST adds these per *phase* and warns
    that they "do not fully support problems with vapoil or disgas active".
    PyMRST converts the boundary flux to a reservoir volumetric rate and hands
    it to the same **component** machinery the interior flux uses (Phase 40),
    so dissolved gas crosses a boundary exactly as it crosses a face.

    Validated on three cases with **no wells at all** - pressure in/pressure
    out, flux in/pressure out, and a source and sink with no boundary - all
    matching MATLAB at **machine precision** (1.5e-16 to 7.6e-15).~~

42. ~~Equilibrium initialisation (`ad_core`, Phase 42). Every initial state in
    this port was set by hand - a uniform pressure and a uniform saturation,
    which is not a reservoir. A real one sits in **gravity-capillary
    equilibrium**: the fluids have had geological time to sort themselves by
    density. Start anywhere else and the first thing a simulation does is fall
    towards this state, which is not the answer you asked for.

    Ported from `initializeEquilibriumPressures` /
    `initializeEquilibriumSaturations` / `initStateBlackOilAD`: integrate
    `dp/dz = g rho(p)` for the reference phase from the datum (an *ODE*, not a
    formula - the density depends on the pressure being solved for), anchor
    every other phase at its contact, then get the saturations by **inverting
    the capillary curve**. With no capillary pressure that inversion degenerates
    to a step - which is to say a sharp fluid contact is not a simplification
    bolted on, it is what a capillary transition zone becomes when `pc -> 0`.

    **The trap, and it is a good one.** A flow model has *one* pressure
    variable. Which phase's pressure should it carry? The reference phase
    (oil) - except where oil is **immobile**. In the water leg below the contact
    there is no oil at all, so the oil pressure there is a fiction: an
    extrapolation of a phase that is not present. Carry it, and the water sees a
    gradient of `rho_oil g` instead of `rho_water g`, and flows. The state looks
    perfectly hydrostatic and then falls over the moment you run it. MRST sets
    the pressure to the dominant *mobile* phase's where the reference phase is
    immobile, and now so does PyMRST.

    Matches MATLAB at **machine precision** (1.8e-16) on both a sharp contact
    and a capillary transition zone - phase pressures, saturations, and the
    pressure the model carries. And the test that actually matters: a state
    called "equilibrium" must not move. Run it 300 days with no wells, no
    boundary and no sources, and it changes by **exactly zero**.~~

43. ~~**CPR: a linear solver that scales** (`ad_core`, Phase 43). Every flow
    model in this port ended in `scipy.sparse.linalg.spsolve` - a sparse
    *direct* solve - and that was the ceiling on field size. It is a real
    ceiling: measured, its cost grows like **n^1.9**, from 0.35 s at 16k cells
    to **17.6 s per Newton iteration** at 98k. No amount of faster assembly
    fixes that.

    CPR (Constrained Pressure Residual) works because a reservoir Jacobian is
    not one problem but two glued together. The **pressure** part is elliptic -
    it is diffusion, it couples the whole field at once, it is what makes the
    direct solve expensive, and it is exactly what algebraic multigrid is built
    for. The **saturation** part is hyperbolic - transport, local, a cell at a
    time - and a cheap incomplete factorisation handles it. So: extract a
    pressure system, solve it with AMG, mop up the rest with ILU, wrap the pair
    in GMRES.

    The decoupling `W` is one line - a weighted sum of each cell's phase
    equations, with weights `1/b`, which cancels the saturation dependence
    exactly (surface mass is `pv b s`, so `sum_i (1/b_i) pv b_i s_i = pv`). It
    works: the decoupled pressure system comes out a textbook **M-matrix** -
    positive diagonal, diagonally dominant, no positive off-diagonals - which is
    checked, not assumed, and AMG converges on it at ~0.3 per V-cycle.

    **The result is that the iteration count stops caring how big the problem
    is** - 13 to 24 GMRES iterations across a **200x range** in size. That flat
    line is the whole point; it is what a working multigrid preconditioner looks
    like, and it is what removes the ceiling:

    | cells | unknowns | direct | CPR | GMRES its | speedup |
    |---|---|---|---|---|---|
    | 2,000 | 4k | **0.01 s** | 0.03 s | 15 | 0.4x |
    | 16,000 | 32k | **0.35 s** | 0.36 s | 13 | 0.9x |
    | 50,000 | 100k | 4.6 s | **1.5 s** | 19 | 3.1x |
    | 98,000 | 196k | 17.8 s | **5.4 s** | 17 | 3.3x |
    | 200,000 | 400k | *(hours)* | **19.7 s** | 24 | - |
    | 432,000 | 864k | *(hours)* | **61.2 s** | 22 | - |

    **CPR is slower than the direct solve below ~30k unknowns**, and that is
    stated rather than buried: the Ruiz sweeps, the ILU and the AMG hierarchy
    all have to be paid for before the first iteration, and on a small problem
    there is nothing to amortise them against. `select_linear_solver` therefore
    picks the direct solve below the crossover and CPR above it, the way MRST's
    `selectLinearSolverAD` does.

    **Three fixes were load-bearing, and only one of them was the one I
    expected.** Drop any and CPR is either slower than what it replaces or fails
    outright:

    1. **Equilibrate first.** A reservoir Jacobian is badly scaled twice over -
       its rows mix units (a well's control equation is not a cell's mass
       balance) and so do its columns (a pressure unknown is ~1e7 Pa beside a
       saturation of ~1). On the raw matrix `spilu` is not merely inaccurate but
       *unstable*: it returned a factor that **amplified** the residual
       10,000-fold.
    2. **Order the unknowns cell-major.** The Jacobian is assembled blocked by
       variable - all pressures, then all saturations - which puts a single
       cell's strong pressure-saturation coupling far off the diagonal, where an
       incomplete factorisation cannot see it. Interleaving to `p1,s1,p2,s2,...`
       is the difference between a factor that is exactly singular and one that
       works.
    3. **`permc_spec="NATURAL"`, and this one is a trap.** SuperLU defaults to
       COLAMD, which minimises fill for a *complete* factorisation - and in
       doing so scrambles precisely the grid locality an *incomplete* one lives
       on. With COLAMD the factor was singular at any tolerance worth using. The
       symptom reads as "the ILU is too weak", and the obvious cure - more fill
       - produces a **48x-fill** factor costing as much as the direct solve that
       *still* needed thousands of GMRES iterations. With NATURAL the same
       problem takes **1.1x fill and 19 iterations**.

    Fix (2) is the one worth being embarrassed about. I diagnosed it correctly,
    tested it, saw it change nothing, and concluded I was wrong - but it changed
    nothing only because COLAMD was re-permuting the matrix anyway, so both
    orderings were equally destroyed. It is load-bearing after all: under
    NATURAL ordering, the variable-blocked layout is singular in *every*
    configuration tried.

    **Validated against MATLAB the only way a linear solver can be**: it must
    not change the answer. Both cases that were already validated against MRST -
    the oil-water quarter-spot and SPE 1 - are replayed through CPR and must
    still clear *the same tolerances the direct solve cleared*, not a bar
    lowered to fit. They do: oil-water to **2.3e-14**, SPE 1 to **2.4e-11**, and
    against the direct solve run side by side, to **1.6e-16** and **1.4e-15**.
    Both need the **identical Newton count** (33 and 161), which is the evidence
    that the linear solves are genuinely converged rather than merely tolerable -
    a solver that is subtly too loose does not crash, it quietly returns a
    slightly wrong Newton update and lets the nonlinear solver absorb it.

    `pyamg` is an **optional** dependency (`pip install pymrst[amg]`). Without
    it the pressure stage falls back to an incomplete factorisation, which still
    beats a direct solve but does not scale the way multigrid does. The rest of
    the port needs nothing but numpy and scipy, and so does this.~~

44. ~~**Adaptive timestep selection** (`ad_core`, Phase 44). Until now this port
    took whatever timestep you handed it and halved it when Newton failed. That
    is enough to be *correct* and it is a bad way to run a schedule: a step
    small enough to survive the hard patches - a well opening, a front arriving
    - is far smaller than the rest of the schedule needs, so you pay for the
    worst moment at every moment; and a step chosen to be efficient fails
    exactly at those patches, where halving is a blunt recovery because it
    throws the work away and does not *remember*. The next report step walks
    into the same wall.

    Ported from `ad-core/timesteps`: `SimpleTimeStepSelector` (the base - hard
    limits, a leash on how fast `dt` may change, ramp-up when the controls
    change), `FactorTimeStepSelector`, `IterationCountTimeStepSelector` (target
    a Newton iteration count - the cheapest honest measure of how hard a step
    was, driven to a set point by what is essentially a PID controller),
    `StateChangeTimeStepSelector` (Cao, 2002 - target a change in the
    *solution*, because iteration count says how hard a step was to solve and
    nothing about how accurate it was: a step can converge in three iterations
    and still push the front half a cell), and `rampup_timesteps`.

    **Matches MATLAB bit for bit** (0.0, exactly) on all six comparisons. A
    selector is a *controller*, so it is validated the way a controller should
    be: driven with a fixed synthetic history - "this step took 3 Newton
    iterations, this one took 12" - with every timestep it picks recorded. That
    pins the control law independently of any model, and it exercises both
    branches, including the one-point `restart` path that exists because a
    controller must not learn from a step it did not choose.

    **And the honest part: a selector is not free.** On a schedule that already
    works it only subdivides steps that did not need subdividing, and it
    *loses* - 52 Newton iterations against 47 on an easy two-year waterflood.
    It earns its keep where a fixed schedule is in real trouble:

    | 4 x 180-day steps, deliberately too coarse | Newtons | time |
    |---|---|---|
    | fixed steps, halve on failure | 229 | 8.7 s |
    | `IterationCountTimeStepSelector` | **159** | **4.5 s** |

    Note that "the fixed schedule takes cuts" is *not* on its own enough to
    make a selector pay - at a gentler injection rate the same case cuts three
    times and halving still wins, because those cuts are cheap. The selector
    wins when the thrashing is expensive. The test asserts that premise rather
    than assuming it, so if the case ever stops being hard the test says so
    instead of quietly comparing two easy runs.~~

45. ~~**A local stage that is nearly free** (`ad_core`, Phase 45). The next item
    on this list was MRST's `DiagonalAutoDiffBackend` - block-diagonal Jacobians,
    a faster assembly. **I profiled it first and the premise was false.** With
    CPR in place, assembly is 2-9% of a Newton iteration (0.13 s against 5.48 s
    at 98,000 cells). Making it *free* would buy 2% at field scale. So it was
    not built, and the profile was followed instead - which pointed somewhere
    much better.

    Inside CPR, the **ILU build was 91%** of the solve (5.2 s of 5.7 s at 98,000
    cells); GMRES itself was 7%. `scipy`'s `spilu` is simply slow. But the local
    stage does not have to be an ILU: after the pressure stage the remaining
    residual is *local*, so invert each cell's `n_phases x n_phases` diagonal
    block and nothing else. It is a batched `numpy` inverse - no factorisation,
    no fill, no ordering to get wrong - and on a two-phase problem it needs **the
    same number of GMRES iterations** as the ILU.

    | 2-phase | direct | CPR (ILU) | **CPR (block-Jacobi)** | its |
    |---|---|---|---|---|
    | 16,000 cells | 0.35 s | 0.36 s | **0.08 s** | 13 |
    | 50,000 cells | 4.6 s | 1.50 s | **0.28 s** | 20 |
    | 98,000 cells | 17.8 s | 5.33 s | **0.52 s** | 17 |
    | 200,000 cells | *(hours)* | 19.4 s | **1.29 s** | 24 |
    | 432,000 cells | *(hours)* | 61.1 s | **2.77 s** | 22 |

    **35x the direct solve at 98,000 cells, and 22x the ILU-CPR it replaces.**
    The crossover moves with it: CPR now overtakes `spsolve` at ~**5,000**
    unknowns rather than 50,000, so `select_linear_solver` carries two
    thresholds - collapsing them into one would route every mid-sized two-phase
    problem to a direct solve that is 4x slower.

    **And then it fails.** On three-phase live oil - a gas cap over an oil leg,
    `rs` switching with saturation - the same block-Jacobi stage does not
    converge *at all*: residual 3.5 after 120 iterations, where the ILU converges
    in **one**. The cell-diagonal block cannot see the gas-dissolution coupling.
    A 15x speed-up that silently diverges on the three-phase case is not a
    speed-up, so: `local_stage="auto"` uses block-Jacobi only where it is
    *measured* to work, and - because an empirical rule from two data points is
    exactly the sort of thing that is wrong on the case nobody tried - a GMRES
    failure **falls back to the ILU and re-solves**. The answer is never allowed
    to depend on the guess.

    Still matches MATLAB exactly (Phase 43's comparison, re-run: oil-water and
    SPE 1, same tolerances, identical Newton counts).~~

46. ~~**Following the profile, twice** (`ad_core`, Phase 46). With the ILU gone,
    the bottleneck moved again - so a full field run was profiled end to end
    rather than guessed at, and the answer was **the AMG application: ~80% of a
    CPR solve**. Three plausible explanations were tested and **two of them were
    wrong**, which is worth recording, because the wrong ones were the ones that
    sounded best:

    - *"`ml.solve` has bookkeeping overhead."* Marginal - 0.21 s against 0.24 s.
      The V-cycles genuinely cost what they cost.
    - *"The decoupling weights are stale - MRST computes them cell-wise and this
      port uses the field-mean pressure."* It turned out MRST's `'simple'`
      weighting **also uses `mean(state.pressure)`**, so the port was already
      faithful; and because `b` varies by under 1% across the field here,
      cell-wise weights bought 61 -> 52 iterations and MRST's rigorous
      **true-IMPES** weights (`pv/rho`, cell-wise) bought **nothing at all**.
    - *"GMRES is restarting - 60 iterations exceeds `restart=50`, and restarted
      GMRES stalls."* Flatly false: restart 50, 120 and 300 all give **66
      iterations**.

    What was actually true is counter-intuitive, and it is a one-line change:
    **one AMG V-cycle beats two.** A stronger pressure stage really does buy
    fewer outer iterations - and not enough of them to pay for itself:

    | V-cycles | GMRES its | time | vs direct |
    |---|---|---|---|
    | **1** | 66 | **0.44 s** | **40x** |
    | 2 (was the default) | 61 | 0.66 s | 27x |
    | 3 | 51 | 0.79 s | 22x |

    Nothing about that is visible without a profile, because every individual
    choice looks reasonable on its own - which is exactly why it is now pinned
    by a test that fails if two cycles ever become worth it.

    One more thing the profile showed, which changes how the earlier numbers
    should be read: **the direct solve gets dramatically worse once a front
    develops.** At 98,000 cells `spsolve` costs 17.8 s on the initial state and
    **66 s** mid-simulation - so the speed-ups quoted from single initial-state
    Jacobians were, if anything, understating the case.

    **The end-to-end result** - whole simulations, not single solves, which is
    the only number that means anything:

    | 5 x 60-day steps | direct | **CPR** | Newtons | speedup | agrees to |
    |---|---|---|---|---|---|
    | 16,000 cells | 24.3 s | **5.2 s** | 26 | 5x | 1.2e-13 |
    | 54,000 cells | **425.6 s** | **18.9 s** | 26 | **23x** | 5.2e-14 |
    | 98,000 cells | *impractical* | **34.3 s** | 26 | - | - |
    | 250,000 cells | *impractical* | **103.1 s** | 27 | - | - |

    The Newton counts are **identical** and the states agree to 1e-13, so this
    is the same simulation, not a cheaper approximation of it. A quarter-million
    cells now runs in under two minutes where the direct solve would take hours.

    Still matches MATLAB exactly (Phase 43's comparison re-run, identical Newton
    counts).~~

47. ~~**Vaporised oil (`rv`) - black-oil is now complete** (`ad_blackoil`,
    Phase 47). Black-oil has *two* dissolutions and this port only ever had one.
    `rs` is gas dissolved in oil; `rv` is oil **vaporised into the gas**. Without
    it a gas cap is dry, a gas-condensate reservoir cannot be modelled at all,
    and gas produced from a volatile oil arrives at the surface carrying no
    liquid. It is not an exotic option - it is half the model.

    What it costs is a **third state per cell**. The hydrocarbon primary
    variable `x` already meant two things; now it means three, and MRST encodes
    which by `status = oil_present + 2*gas_present`:

    | status | phases present | `x` is |
    |---|---|---|
    | 1 | oil, no free gas | `rs` |
    | **2** | **gas, no free oil** | **`rv`** - the state this port never had |
    | 3 | both | `sG` |

    Ported: `OilComponent` (the exact mirror of `GasComponent` - oil in the oil
    phase *and* `rv·b_g` in the gas, which the conservation equation cannot tell
    apart from dissolved gas, and that symmetry is the whole argument for the
    component framework); the three-way switch and its reconstruction; the flash
    (`computeFlashBlackOil`) with both phase appearances; `rv` in the density,
    so a wet gas is heavier than a dry one and segregates differently; and the
    well's **dissolution matrix**, whose determinant `d = 1 - rs·rv` is why
    `rs·rv < 1` is a physical requirement rather than a modelling convenience -
    at `rs·rv = 1` oil and gas have become the same fluid.

    **Matches MATLAB at machine precision** on a gas-condensate column - a gas
    cap with *no oil phase at all* over an undersaturated oil leg - in two
    cases, because one would not have been enough: a **steady** one where the
    cells stay in status 2 and 1 for every step (pressures to **9.7e-16**), and
    a **depletion** where gas breaks out of solution and cells cross into status
    3 (**3.9e-9**). The status history - an *integer* decision about which
    variable each cell is solving for, where "close" is meaningless - is
    **identical** in both. And the check no reference can make: seal the box, no
    wells, no boundary, and a wet-gas reservoir must not move. It moves by
    **exactly zero**.

    **Two bugs fell out, and one of them was old.** `update_state` applied the
    `dp_max_rel` pressure cap *only* in the two-phase branch - the three-phase
    model had been taking the raw Newton increment since it was written. SPE 1
    never noticed, because its pressure moves gently enough that a 20% limit
    never binds; a nearly incompressible oil leg asks for a 70 bar first step,
    overshoots, and oscillates instead of converging. And `_state0_props` copied
    the previous state's `rs`/`rv` verbatim where MRST *reconstructs* them from
    `status` - so the accumulation term differenced two states that did not mean
    the same thing.

    **A finding about MRST worth stating.** MRST's flash only lets gas nucleate
    once `rs > rsSat·(1 + sqrt(eps))`. The dead-band stops a cell chattering in
    and out of the two-phase state on round-off - but it also means a cell
    sitting *exactly* on its bubble point, which is what a depleting oil leg
    produces in quantity, can be asked by Newton for a gas increment smaller
    than the dead-band, have it clipped straight back, and **stall**. The
    residual parks at ~1e-7 and will not go lower however long you iterate. MRST
    never sees it because its default `toleranceCNV` is **1e-3**; PyMRST, asking
    for 1e-8, walks straight into it. The tolerance is what is wrong there, not
    the physics: **1e-8 is not a tolerance this discretisation can always
    reach**, and PyMRST's default is optimistic.~~

48. ~~**Fetkovich analytic aquifers** (`ad_core`, Phase 48). A reservoir usually
    sits in water, and gridding that water is wasteful: it is enormous, nothing
    interesting happens in it, and all the reservoir feels is the pressure
    support at its edge. An *analytic* aquifer is one lumped tank of known
    pressure and volume, connected to boundary cells: it pushes water in as the
    field is produced and its own pressure falls in response.

    Ported from `AquiferModel.m` (ECLIPSE `AQUFETP`/`AQUANCON`): the influx
    `q = alpha·J·(p_aq − p_w + rho_w·g·dz)·coef`, the Fetkovich relaxation
    `coef = (1 − exp(−dt/Tc))/(dt/Tc)` with `Tc = C·V/J`, and the
    after-convergence update that drains the tank.

    **Validated twice, because one way would not have been enough.**

    - **The formulas, in isolation** (Phase 48). The dump fixes the water
      pressure and shrinkage *MRST itself evaluated* at the connections and
      PyMRST replays its own arithmetic on the identical numbers - so what is
      compared is the physics, not the two codes' fluid libraries. Influx at
      `dt = 0` and at a large `dt`, and the whole drawdown trajectory: **all
      bit-for-bit exact (0.000e+00)**. The large-`dt` sample exists to force the
      relaxation coefficient below 1 (it lands at 0.85 and 0.78), because a test
      in which `coef == 1` would pass without ever exercising it.
    - **The coupling, in a running simulation** (Phase 48b). The formulas being
      right says nothing about the *plumbing*, and the plumbing is where this
      goes wrong: flip the sign and the aquifer drains the reservoir instead of
      supporting it; hold it explicit at the wrong state and it drains at the
      wrong rate. So a real oil-water field, a producer pulling it down and an
      aquifer holding it up over twenty steps - pressures to **4.4e-11**,
      saturations to **3.4e-09**, and the aquifer's own pressure and volume to
      **1.2e-12** and **2.7e-15**. Plus the two checks no reference can make:
      the tank gives up **19.6 bar** (a decoration would give up nothing), and
      it stays *above* the cells it feeds at every step (a sign error flips
      that).

    **What the coupled comparison cost, and it is worth knowing.** Two things
    that were not aquifer bugs at all had to be cleared out of the way first:

    - PyMRST's connection pressure drop is a single hydrostatic column per
      perforation - exact for a one-cell well, an approximation for a many-cell
      one, and a *known* gap (it does not do MRST's wellbore-topology sum). With
      a three-cell producer it polluted the comparison at the **1e-3** level and
      looked exactly like an aquifer error. The well is one cell now, and the
      comparison measures the aquifer.
    - **MRST's default tolerances leave a real residual in its own answer.**
      `toleranceCNV = 1e-3` stops the Newton loop long before the discrete
      equations are solved; PyMRST drives to `1e-8` and is therefore the *more*
      converged of the two, and the leftover difference - **1e-5**, growing over
      the run - reads as a modelling disagreement when it is nothing but MRST's
      slack. Tightening MRST to 1e-10 collapsed it to **1e-11**. Any comparison
      that claims better than ~1e-5 against stock MRST tolerances should be
      treated with suspicion.~~

49. ~~**Multi-perforation wells - the wellbore column** (`ad_core`, Phase 49).
    Not a missing feature. A path that was **wrong, and that nothing had ever
    tested.**

    Every well in every MATLAB comparison this port had run had **one
    perforation**. SPE 1: one. Norne: one. The oil-water quarter spot, the
    gas-condensate column, the aquifer case: one each. So the part of the well
    model that only exists when a well has *more* than one - the wellbore column
    - had never once been checked against MRST. And every real well has dozens.

    A well is a chain of nodes: the bhp datum, then each perforation. The fluid
    in the segment above a perforation is everything flowing *past* it - the
    cumulative stream from every perforation below - so each segment carries its
    own mixture, its own density, and

        cdp_k = sum_{j<=k} rho_j g (dZ_j - dZ_{j-1})

    with `rho_j` that segment's cumulative mass flux over its cumulative volume
    flux. PyMRST computed a **single density times the total depth**. For one
    perforation those are the same thing *exactly* - which is why every
    validation passed for months. For three they are not: a three-cell producer
    disagreed with MRST at the **1e-3** level, which I first met while building
    the aquifer test and stepped around by shrinking the well to one cell.

    **Ported the wrong function first, and the data caught it.** MRST has two:
    the classic facility's `updateConnectionPressureDrop` (a `wb2in` solve on
    *surface* rates, with a zero-flux guard that gives an unflowed well a 50/50
    phase mixture) and `updateConnectionPressureDropState`, which is what
    `GenericFacilityModel` - and therefore every model here - actually calls. I
    implemented the first. It broke the aquifer comparison from 4e-11 to 5e-4,
    which was the clue: the *old* code had already been an exact match to the
    generic one for a single perforation, so any "fix" that changed a one-cell
    answer had to be wrong.

    **And then a bug the fix itself exposed.** `_state0_props` builds a reduced
    previous state - and it dropped `well_flux`. So the solver never saw the
    flux history and fell back to the no-history branch on *every* step, while
    the validator (reading the full state) reported the right column. The
    recorded cdp was not the one being used. It surfaced only because forcing
    MRST's own cdp into PyMRST gave **1.54e-15** while PyMRST's own gave 3.7e-5
    - two numbers that cannot both be true, which is exactly the kind of
    contradiction worth chasing.

    **Validated on two vertical wells, five perforations each**, with a heavy
    density contrast (water 1050, oil 700) so that *which* fluid sits in *which*
    segment matters. The wellbore column itself is compared, not just its
    consequences: injector cdp **exactly** (0.0), producer **4.8e-16**,
    pressures **1.5e-15**, rates **1e-15**. And the check no reference can make:
    the producer's column implies segment densities of **776, 792, 811, 840,
    901 kg/m3** - the stream gets heavier as it collects water on the way up -
    where a single-density column can only ever produce identical ones.~~

50. ~~**Compositional phase density and viscosity** (`compositional`, Phase 50).
    The flash (Phase 13) answers what an equation of state is *for* - given a
    pressure, a temperature and an overall composition, which phases exist and
    what is in them - and stops at compositions and compressibility factors. A
    flow model cannot use any of that: Darcy's law wants a **density** and a
    **viscosity**, and neither is in a flash result. That missing half is why
    the EOS had sat unused since it was ported.

    `compositional/properties.py` ports `CompositionalPropertyModel`: the density
    `rho = M(x)/(Z R T/p)` with the optional **volume shift** (a per-component
    correction to Peng-Robinson's poor liquid volumes, legitimate precisely
    because it cancels out of the fugacity condition and so moves the density
    without moving the equilibrium), and the **Lohrenz-Bray-Clark** viscosity -
    which is written in Rankine, psia and centipoise and whose constants mean
    nothing in SI, so the units are converted at the boundary and the
    correlation is left as Lohrenz wrote it. Fed **MRST's own flash outputs**
    across 84 liquid/vapour/two-phase conditions, with and without a volume
    shift, PyMRST matches MATLAB at machine precision: density to **1.7e-16**,
    viscosity to **2.7e-15**.~~

51. ~~**Flash -> flow properties** (`compositional`, Phase 51). The last analytic
    step between a flash and a compositional flow model. A per-component
    conservation equation does not read a flash directly; it reads mass fractions
    and phase saturations, and `computeTwoPhaseFlowProps` is where MRST builds
    them. `compositional/flow_props.py` ports it:

    - **mass fractions** `xM`, `yM` (`getMassFraction`): the flash speaks in
      moles, the accumulation term `pv/dt(rhoO sO xM - ...)` is written in mass,
      so `w_i = M_i x_i / sum_j M_j x_j`.
    - **phase saturations** `sL`, `sV` (`computeSaturations`): the flash gives a
      *molar* liquid fraction `L`; a saturation is a *volume* fraction. With no
      volume shift the phase volumes are `L Z_L` and `(1-L) Z_V` (the `R T/p` of
      the molar volume cancels between the phases), so
      `sL = L Z_L / (L Z_L + (1-L) Z_V)`. With a shift they go through the shifted
      molar density instead - which moves the volume split while leaving the
      molar flash untouched.
    - the `(p,T,z) -> {xM, yM, sL, sV, rhoL, rhoV, muL, muV}` pipeline the flow
      model will call once per cell per Newton iteration.

    Replayed on **MRST's own flash** across the same 84 conditions, PyMRST
    matches MATLAB to machine precision: mass fractions **exact (0.0)**,
    saturations to **8.7e-19**, including the volume-shift saturation split
    (which is verified to actually move, so the shifted check is not vacuous).

    Every prerequisite for an overall-composition flow model is now in place -
    flash (13), properties (50), flow props (51), differentiated flash (52), and
    the assembled flow model itself (53). Component-aware wells are next.~~

52. ~~**Differentiating the flash** (`compositional`, Phase 52). The flash solves
    for the phase split on *values* - there is no derivative in the answer. But a
    fully-implicit compositional model differentiates against pressure and the
    overall composition `z`, and every property it assembles is a function of the
    flash. So it needs `dL/dp, dL/dz, dx/dp, dx/dz, dy/dp, dy/dz`, and the flash
    is an *implicit* function of `(p,T,z)` - defined by the equilibrium
    conditions, not a closed form. This is the one genuinely hard piece of the
    compositional port, and it is what MRST's `getPhaseFractionAsADI` exists to
    do.

    `eos_ad.py` ports it via the **implicit function theorem**. With
    `F(s;u)=0` the equilibrium system - secondary `s=(L,x,y)`, primary
    `u=(p,T,z)`, equations the component balances, equal fugacity and phase
    closure - the sensitivities are `ds/du = -(dF/ds)^{-1}(dF/du)`. The two
    Jacobians are obtained by evaluating `F` twice under PyMRST's own
    forward-mode AD (once seeding `u`, once seeding `s`); because a cell's
    equilibrium couples only that cell's own variables, both are block-diagonal
    by cell, so the global sparse LU MRST does becomes a batch of tiny dense
    `(2n+1)x(2n+1)` solves. The subtlety the value-only flash sidesteps is that
    the compressibility `Z` is a *root of the cubic* and carries no automatic
    derivative; MRST's `setZDerivatives` differentiates the cubic implicitly
    (`dZ = -(dE2 Z^2 + dE1 Z + dE0)/(3Z^2 + 2E2 Z + E1)`) and so does this. The
    A/B mixing rule, cubic and fugacity are the same formulas `eos.py` already
    matched MRST on - re-expressed to carry derivatives.

    **Validated two independent ways.** Against MRST's own
    `getPhaseFractionDerivativesPTZ`, fed MRST's exact flash: `dL`, and all
    three `dx_i` and `dy_i`, match to **~1e-15 relative** across 35 two-phase
    cells. And against **finite differences of the flash itself** (a check that
    needs no reference at all): the pressure derivative directly, and the
    composition derivatives as the feasible direction `d/dz_j - d/dz_n` that
    respects `sum z = 1` - the exact combination the flow model uses. Two
    different questions - "does it match MATLAB's derivative routine" and "is it
    actually the derivative of the flash" - and both say yes.

    The overall-composition flow model now has every part it needs, each ported
    and MATLAB-validated on its own; what is left is assembling them into the
    `equationsCompositional` residual and its component wells.~~

53. ~~**Overall-composition flow model** (`compositional`, Phase 53). The
    assembly: every validated piece wired into a fully-implicit simulator.
    `compositional/model.py` ports `equationsCompositional` (no-water branch):
    per-component **mass** conservation

        (pv/dt)(rhoO sO xM_i - ... + rhoG sG yM_i - ...) + div(rOvO upstr(xM_i) + rGvG upstr(yM_i)) = 0

    with primaries `(p, z_1..z_{n-1})` and `z_n = 1 - sum z_i`. Each Newton
    iteration calls the ADI branch of `computeTwoPhaseFlowProps` through
    `flash_ad`, `phase_compressibility_ad`, `flow_props`, and `properties` - the
    whole chain validated in Phases 13/50/51/52.

    **Validated end-to-end against `simulateScheduleAD`.** A closed 1D column
    (10 cells, no wells, no-flow boundaries) started off a pressure gradient and
    left to equilibrate - accumulation and flux both active, no source to hide an
    error. Five steps of 20 days; MRST run with tightened tolerances (1e-10, not
    the 1e-3 default that stops before the equations are solved). PyMRST matches:
    pressure to **3.7e-4 bar** (a small systematic offset from step 1, not
    drift), composition to **7.6e-7**, saturation to **3.7e-6**. Component mass
    conserved to **1.5e-11**. Independently: recomputing `sO` from MRST's own
    `p,z` matches MRST's stored saturation to **5e-10** - the saturation formula
    is right; the trajectory gap is the tiny `p,z` offset. Module self-tests add
    a finite-difference Jacobian check, closed-domain mass balance, and
    single-phase stability.

    **Still out of scope here:** component-aware wells, water, gravity,
    capillary pressure. The closed-domain model is the validated core; wells are
    the next slice.~~

54. ~~**Component-aware compositional wells** (`compositional`, Phase 54). The
    well slice on top of Phase 53: standard MRST surface-rate well primaries
    (`qOs`, `qGs`, `bhp`) plus per-component mass sources from
    `getExtraWellContributions` - injection at the well-stream mass fraction
    (`W.components`), production at local `xM`/`yM`. `Well.components` added to
    `params/wells_and_bc/types.py`.

    **Validated end-to-end against `simulateScheduleAD`.** A 1D methane-rich gas
    injector (rate control, `W.components = [0.9, 0.05, 0.05]`) and BHP producer
    (10 cells, 5×30 days). PyMRST matches: pressure to **7.9e-6 bar**, composition
    to **6.9e-7**, saturation to **1.0e-6**, well rates to **2e-9 m³/s**, BHP to
    **7.9e-6 bar**. Module self-tests add injector/producer convergence and
    closed-domain mass balance (Phase 53 regression unchanged).~~

55. ~~**Compositional gravity** (`compositional`, Phase 55). The no-water
    overall-composition model now drives each phase flux with the full
    potential `grad(p) - avg(rho_phase) * g * grad(z)`, keeping the Phase 53/54
    no-gravity paths as the default.

    **Validated end-to-end against `simulateScheduleAD`.** A closed vertical
    oil/gas column (10 cells, 5×20 days, gravity on) segregates under gravity.
    PyMRST matches: pressure to **2.4e-4 bar**, composition to **1.6e-6**, oil
    and gas saturations to **8.4e-6**. Component mass is conserved and methane
    enriches upward (`0.7516` top vs `0.2676` bottom). Module self-tests add a
    gravity-column segregation check; Phases 53 and 54 remain unchanged.~~

56. ~~**Compositional water, closed-domain slice** (`compositional`, Phase 56).
    The overall-composition model now supports `model.water = true` without
    wells: `sW` is a primary variable, oil/gas saturations are scaled by
    `(1-sW)`, and a water conservation equation/flux is assembled beside the
    hydrocarbon component equations.

    **Validated end-to-end against `simulateScheduleAD`.** A closed 1D
    water/oil/gas case (10 cells, no wells, no gravity, no capillary pressure,
    5×20 days) starts from a pressure and water-saturation gradient. PyMRST
    matches: pressure to **3.7e-4 bar**, water saturation to **1.6e-8**,
    composition to **7.5e-7**, oil/gas saturation to **3.2e-6**. Water and
    hydrocarbon component masses are conserved. Module self-tests add a water
    conservation check; Phases 53-55 remain unchanged.~~

57. ~~**Compositional water wells** (`compositional`, Phase 57). Water-active
    compositional states now carry `qWs` beside `qOs`, `qGs`, and `bhp`; the
    well model handles water perforation rates, water conservation sources,
    oil/gas component sources, and rate/BHP controls.

    **Validated end-to-end against `simulateScheduleAD`.** A water injector at
    fixed surface rate and a BHP producer (10 cells, `model.water = true`,
    5×30 days, gravity/capillary off). PyMRST matches: pressure to
    **1.1e-5 bar**, water saturation to **3.4e-7**, composition to **1.3e-6**,
    oil/gas saturation to **2.1e-6**, well rates to **1.5e-9 m³/s**, and BHP to
    **1.1e-5 bar**. The injector water rate stays on control and the injector
    cell water saturation rises from `0.6220` to `0.9289`.~~

58. ~~**Compositional capillary pressure** (`compositional`, Phase 58). The
    overall-composition model now accepts optional `pc_ow` and `pc_og` callables
    and uses MRST's phase-pressure convention in the flux potentials:
    `pW = p - pcOW(sW)`, oil at `p`, and `pG = p + pcOG(sG)`.

    **Validated end-to-end against `simulateScheduleAD`.** A closed water/oil/gas
    case (10 cells, no wells, no gravity, 5×20 days) uses
    `pcOW(sW)=pc0*(1-sW)`. PyMRST matches: pressure to **2.3e-4 bar**, water
    saturation to **1.3e-10**, composition to **5.8e-10**, and oil/gas
    saturation to **1.4e-6**. The capillary pressure visibly moves water
    (`||delta sW|| = 3.53e-3`).~~

59. ~~**Compositional pressure reduction** (`compositional`, Phase 59). The
    `opt.pressure` assembly slice is now present: PyMRST computes numerical
    partial-volume weights from the accumulation Jacobian and collapses the
    water/component conservation equations into MRST's weighted pressure
    equation.

    **Validated against MRST's `PressureOverallCompositionModel`.** A direct
    one-step equation assembly (8 cells, no wells, no water) compares the
    partial-volume weights, reduced pressure residual, and pressure-Jacobian
    block. PyMRST matches to **2.6e-2** in weights, **4.9e-2** in residual, and
    **3.5e-7** in the pressure Jacobian block. This is the reduced equation
    path, not yet a full sequential pressure-only Newton solve.~~

60. ~~**Compositional pressure-only step** (`compositional`, Phase 60).
    `OverallCompositionModel.pressure_step()` now solves the reduced
    `opt.pressure` Newton system with transport variables frozen. It assembles
    with full composition derivatives so the partial-volume weights are correct,
    then keeps only pressure plus well variables active, mirroring MRST's
    pressure model.

    **Validated against MRST's `PressureOverallCompositionModel`.** A direct
    no-well pressure-only Newton step (8 cells, no water, no wells) compares the
    solved pressure and frozen composition. PyMRST matches pressure to
    **0.33 bar** and keeps composition identical. The pressure residual in
    PyMRST is solved to `~4e-11`; the remaining offset is the same small
    EOS/partial-volume-weight mismatch seen in Phase 59, amplified through the
    pressure-only solve.~~

There is also a first self-tested no-water/no-well `transport_step()` using
MRST's temporary `sT` primary. It is not counted as MATLAB parity yet; the
direct MRST transport reference still needs a reliable harness.

**Still missing for full compositional parity:** MATLAB-validated transport for
water/well cases and the full sequential pressure/transport wrapper.

**`ad-mechanics` is now complete**, Norne included.

### The one place PyMRST and MRST genuinely disagree on Norne

Everything above matches to 1e-7 from step 1 onward. **Step 0 differs by
3.7e-5**, and the cause is worth stating rather than hiding under a tolerance.

MRST takes each well's wellbore density from its connection fluxes, which are
zero before the well has ever flowed. Its guard replaces those zeros with
`sign*1e-12` - *the same value in every phase* - so a producer that has not yet
produced is modelled as holding a 50/50 mixture (850 kg/m3) rather than the oil
it is actually sitting in (700 kg/m3). Over that well's 62 m of column it is
0.91 bar of connection pressure drop, on the first step only.

PyMRST uses the physical mixture. From step 1 both wells have flowed, both codes
use the real one, and the discrepancy decays: 3.7e-5, 1.8e-6, 4.6e-7, ... 6.3e-8.
That decay is the evidence that it is one initialisation difference and not a
systematic error - which is why `validate_norne.py` asserts the decay rather
than loosening the tolerance.

### Two lessons from a real corner-point grid

Both were bugs in the Norne example, both were silent, and both are the same
mistake - selecting a boundary by *depth* when a corner-point surface is not
flat:

- Norne's top has 40 m of relief, so "the shallowest nodes" is a handful of
  nodes on one high spot, not the surface. Uplift was being read off a single
  node.
- Worse, its base is not flat either, so clamping "the deepest boundary faces"
  clamped **two faces**. The model stayed solvable, converged happily, and
  produced a beautifully plausible dilation field while quietly moving tens of
  metres. Nothing warns you; the residual is small because the equations are
  satisfied - they are just not the equations you meant.

Both boundaries must be picked by the face *normal* (which is what MRST's
direction tags amount to). The mechanics itself was never wrong: with the base
properly clamped, `div(u)` reproduces the closed form `dp/K` to three digits.

Next: full TEDFM fracture workflows and the CMG-export pipeline. `ROADMAP.md` plans the rest
(Phases 25-42) in dependency order.

### A seventh finding: MRST's oil-water poromechanics drops a cell volume

`equationsOilWaterMech.m` computes the mechanically-corrected pore volume as

```matlab
effPorVol = rock.poro .* (G.cells.volumes .* pvMult) + rock.alpha .* mechTerm
```

where `mechTerm` is `div(u)` - a *dimensionless* strain. Its two siblings do
not agree:

```matlab
equationsWaterMech.m:    effPorVol = G.cells.volumes .* (poro .* pvMult + alpha .* mechTerm)
equationsBlackOilMech.m: effPorVol = G.cells.volumes .* (poro .* pvMult + alpha .* mechTerm)
```

So the oil-water model is **missing `G.cells.volumes` on the Biot term**: it
adds a bare strain to a volume. The Biot coupling is therefore too weak by a
factor of the cell volume, and - being dimensionally inconsistent - its
strength depends on the grid's unit of length. On a grid of unit cells the
factor is 1 and the defect is invisible, which is presumably how it survived.

It is not a rounding-level concern. On cells of 500 m³ it moves the pressure
by **10%**. `validate_phase27.py` pins the diagnosis rather than asserting it:
the same case is run on cells of 500 m³ and of 1 m³, and

- on the 500 m³ grid, a correct Biot term disagrees with MRST by 9.9e-2,
  while **putting the cell volume back out reproduces MRST to 2.8e-15** - so
  the missing factor is the whole of the difference, to machine precision;
- on the 1 m³ grid the two formulas coincide, and both match MRST.

PyMRST's `MechBlackOilModel` is correct; pass `mrst_oilwater_bug=True` to
reproduce MRST's number, exactly as `LimAzizShapeFactor` does for its own
MRST defect.

### A correction: the convergence criterion was not MRST's, and said so anyway

Found by being asked a good question - *why are PyMRST's equations in surface
volumes when MRST's are in mass?* - and worth writing down because the premise
of the question turned out to be half wrong and the answer turned out to be a
bug somewhere else entirely.

**The equations were never the problem.** MRST's *classic* black-oil equations
are in surface volumes too - `equationsOilWater.m` has
`water = (1/dt)(pv bW sW - pv0 bW0 sW0)`, no `rhoWS` anywhere - and that is what
Phase 5 ported and validated. Only MRST's *newer* state-function path switched to
a mass basis, so MRST carries both conventions in two code paths. They differ by
one constant per equation, `eq_i^mass = rho_i^s eq_i^surfvol`, which is a row
scaling of the residual and leaves the Newton step exactly invariant:
`(DJ)^-1 (Dr) = J^-1 r`. MRST agrees, and proves it - `getConvergenceValuesCNV`
divides the surface density straight back out (`eq = eq./rhoS`) before testing
convergence, so its own criterion is computed on the surface-volume residual.

**What was wrong was the criterion.** The same MRST function also weights each
phase by `B_avg = mean(1/b)` - the average formation volume factor, which turns a
surface-volume imbalance back into the reservoir volume it actually occupies - and
PyMRST's `get_convergence` did not. It also reads the *current* pore volume from
the state, where PyMRST used the reference one.

That is not cosmetic. On SPE 1:

| phase | `b` | `B_avg = mean(1/b)` |
|---|---|---|
| water | 0.97 | 1.02 |
| oil | 0.60 | 1.72 |
| **gas** | **262** | **0.0032** |

A cubic metre of *surface* gas is not a cubic metre of anything downhole. Without
the weight, PyMRST's gas CNV was **262x larger** than MRST's for the identical
error - so `tol_cnv` was ~262x stricter on gas than on water, and the docstring's
claim that this was "MRST's CNV" was simply false. Nothing was *wrong*: being too
strict is safe, which is exactly why every validation passed and the defect
survived. It just made the tolerance mean something nobody had chosen.

Fixed in all three black-oil models (the graph model and both hand-rolled ones -
they have to agree, or two of them stop in different places). **SPE 1 now
converges in 148 Newton iterations instead of 161**, and still matches MATLAB to
**2e-15**. Every MATLAB validation passes unchanged.

The lesson generalises past this bug: *a criterion that is too strict never
fails, so nothing ever forces you to look at it.*

**And then: "MRST has both - why doesn't PyMRST?"** A fair follow-up, and the
answer is *it does now* - but not the way MRST does it.

MRST's duplication is not a feature it offers, it is a tax it pays. The
`isMass = isa(model, 'GenericReservoirModel')` branch inside
`getConvergenceValuesCNV` exists for no reason other than to ask which of the
two code paths it is standing in and unscale accordingly. Copying two parallel
equation assemblies would have doubled the surface area of every future change
(vaporised oil alone touched five component densities and the well model) to buy
**zero numerical capability**, since the two are a row scaling apart.

But the instinct behind the question is right, because surface volume does not
generalise: a compositional species has no surface volume, and energy is not a
volume at all - so a compositional or thermal model *cannot* be written in this
basis, and both are still unported.

So `ReservoirModel` now takes **`mass_basis`** - one flag, not two code paths.
Six places ask it: the accumulation, the flux, and the four things that put a
component into an equation from outside (wells, boundaries, sources, aquifers),
each of which meters in surface volumes and must be weighed before it can enter
a mass equation. `get_convergence` takes the residual back out again, which is
precisely MRST's `isMass` branch - and is why the two bases do not merely
converge *similarly*, they converge **identically**.

That last claim had been sitting in this port's docstrings, unchecked, for a
long time. It is now a test: the same case - with a well **and** a boundary
condition **and** a source **and** an aquifer, because each is a separate place
to forget the scaling - run in both bases takes the **same 14 Newton iterations**
and agrees to **8.4e-16**. Surface volume remains the default, because it is
what every validated result here was computed in.

### An eighth finding: MRST's flash has a dead-band that caps how far Newton can converge

`computeFlashBlackOil` will only let gas nucleate in an undersaturated cell once
`rs > rsSat*(1 + sqrt(eps))`. The dead-band is deliberate and sensible - without
it a cell would chatter in and out of the two-phase state on round-off.

But it also means a cell sitting **exactly on its bubble point** - which is what
a depleting oil leg produces, in quantity - can be asked by Newton for a gas
increment *smaller than the dead-band*, have `rs` clipped straight back to
`rsSat`, and make no progress at all. The iteration then stalls: the residual
parks at ~1e-7 and does not move however long it runs. Nothing is wrong with the
Jacobian; there is simply no state the flash will accept that reduces the
residual further.

MRST never encounters this, because its default `toleranceCNV` is **1e-3** and it
has long since stopped. PyMRST, whose default is `1e-8`, walks straight into it.
That is not a bug in the port and it is not really a bug in MRST - it is a
property of the discretisation, and the honest conclusion is that **1e-8 is not
a tolerance a black-oil flash can always reach**. `validate_phase47.py` therefore
runs at MRST's own tolerances and says why; PyMRST's defaults remain tighter
because they are usually attainable, but they are optimistic rather than
guaranteed.

### Two traps in coupled poroelasticity, both of which cost real time here

Both are about the *previous* state, and both silently destroy the Mandel
overshoot while leaving Terzaghi looking fine - which is precisely why
Mandel is in the test suite.

- **A stored state must carry its own displacement.** `div(u)` is assembled
  from the free dofs *plus* a `div_rhs` built from the prescribed boundary
  displacements. Mandel's rigid plate moves every timestep, so re-deriving
  the old state's `div(u)` from its free dofs evaluates it against the
  *new* plate position - and the pore-volume change the flow equation
  exists to see cancels itself out. MRST stores `state.u` (the full
  displacement) for exactly this reason; PyMRST now does too.
- **The unloaded state needs an explicit zero `u`**, not an implied one, or
  the slab appears to have been deformed before it was ever loaded.

A third, smaller one: MRST drives Mandel's constant-force plate by rescaling
(`dy = dy * load / force`), which assumes the reaction force is
*proportional* to the plate displacement. It is *affine* - the pressure
supplies an offset - and the slope here is so flat that the rescaling crawls
and never arrives. A secant iteration lands on it in one step.

### A sixth finding: MRST's ad-mechanics under-converges at its defaults

The coupled model first matched MATLAB to only 2e-7 relative - while the
*first two steps* matched to 1e-13. Equal equations, so the drift had to be
a stopping rule, and it was: MRST's default `toleranceCNV = 1e-3` and
`toleranceMB = 1e-7` stop Newton well short of solving the discrete system,
and the leftover residual accumulates across a schedule. Tightening them in
`dump_phase27_reference.m` takes the agreement to 4e-9 in pressure and 9e-10
in displacement. This is the same class of finding as `ad-micp`'s and the
dual-porosity models': the number a default-tolerance MRST run reports is
not the solution of the system it set up.

### A fifth finding: MRST's pebi is fragile on flat boundary triangulations

While choosing Phase-23 validation cases, MATLAB's own `pebi` failed its
`sortCellFaces` closure assertion on a valid Delaunay input (two rows of
obtuse triangles, a 6+5-point strip with 0.3 aspect). The validation cases
were therefore chosen among inputs the reference itself survives; PyMRST's
port reproduces MRST's algorithm faithfully, including whatever that
algorithm cannot handle.

### The one module with no MATLAB reference (besides the adjoint)

`ve_dissolution` is verified by physics, not against MATLAB. MRST's version
lives inside the `CO2VEBlackOilTypeModel` statefunction graph, and porting
that scaffolding to drive a comparison would have cost more than the physics
is worth. Instead the rate law is ported term by term from
`computeDissolutionTransfer` and checked on the invariants that would break if
it were wrong: total CO2 (free + dissolved) is conserved to 4e-8 once
injection stops; the brine saturates at `dis_max` and no further; CO2 cannot
dissolve where there is none to dissolve; and dissolution speeds up with
`dis_rate`. Only the **rate-driven** branch is ported - MRST's `dis_rate = 0`
(instantaneous) is implemented by variable switching, as in black-oil, and is
not here.

One subtlety that the trapping inventory's own mass balance caught: dissolution
eats the *residual* CO2 too, so `sGmax` has to be allowed to **fall**
(MRST: `sGmax = max(sGmax0 - s_depletion dt fac, sG)`). With the usual
`sGmax = max(sGmax0, sG)` the inventory credits residual CO2 that dissolution
has already removed, and the categories stop adding up.

### CO2 trap catchments: a fixed bug worth remembering

`trap_regions` - which trap each cell drains to - now matches MRST **exactly**
(1440/1440 cells) on a caprock without degenerate ties. Getting there turned
up three things that were not obvious:

- The flow graph joins neighbours at **exactly equal** depth in *both*
  directions, so a level patch of caprock is crossed as one connected unit.
  PyMRST used a 1e-9 tolerance for "equal". That is poison: it also joins
  neighbours differing by 1e-13, which are not level at all, and because each
  such edge is bidirectional it lets the catchment search walk *uphill* across
  a divide and flood the wrong trap's region. 183 spurious edges put the
  catchments 5% off MRST. The tolerance has to be zero, and
  `test_flat_tolerance_must_be_exact` now pins it.
- MRST's `top` lists **every** summit inside a trap, not one crest per trap.
- Catchments are seeded from each summit in turn and later ones **overwrite**
  earlier ones, so a trap with two summits gets the last word wherever its
  catchment overlaps another trap's.

On a *symmetric* caprock the two codes still differ (96.5% of cells), and that
one is not fixable: mirror-image cells sit at bit-identical depths in MATLAB
but 2e-13 apart in the Python grid, so MATLAB finds a third summit that Python
does not. A round-off tie on a perfectly symmetric surface.
`validate_phase17.py` checks the asymmetric case exactly and prints the
symmetric one for the record.

Two traps for anyone comparing trap output: **trap labels are arbitrary** in
both codes (compare `trap_regions` only after mapping labels by trap-cell
overlap - comparing the raw integers understates agreement badly), and
`maximizeTrapping` ranks on **bulk** trap volume, not pore volume.

### How the adjoint is verified

The adjoint is the one module with no MATLAB reference, and it does not
need one. A gradient is either the derivative of the forward model or it
is not, and finite differences of that model settle the question: the
adjoint gradient agrees with them to 8e-10 relative. The forward model it
differentiates is itself validated against MATLAB (Phases 4 and 5), so the
chain is closed. `dR/du` and `dR/dx_{n-1}`, the two blocks the adjoint adds
to the forward assembly, are each finite-differenced separately as well,
and the NPV sign convention is pinned by its own test - a flipped sign
still passes a gradient check, so it needs checking on its own.

### A fourth finding: co2lab-ve's 'S table' branch cannot run

`addVERelpermCapillaryFringe`'s `'S table'` variant raises the moment it is
used: `truncate_at_aquifer_bottom` writes `vec_truncated(trunctate)`,
misspelling its own mask, so the branch is dead code. Confirmed by calling it -
it fails with `Unrecognized function or variable 'trunctate'`. PyMRST's
`make_ve_table_h` is that model with the typo fixed, and therefore has no MRST
reference to validate against; the default `'P-scaled table'` branch works and
is what Phase 20 checks, to 4e-16.

### A third finding: MRST's raster reader is off by one at both ends

`readAAIGrid` loses a row of data at each end of the file, and the Atlas grids
depend on it:

- its header loop reads a line and only *then* discovers the line is data, at
  which point it breaks - so the **first data row is consumed and discarded**;
- its data loop runs `while i < nrows`, one iteration short, so the **last row
  is never filled** and stays zero (which it then maps to NaN).

The net effect is a raster shifted one row against the file. PyMRST reproduces
it, because otherwise the grids do not match - but anyone using these grids
should know the formation is displaced 200 m north of where the file says.

Two further traps in the same code, both of which cost real time here: the
top surface must be **sampled** at the raster nodes rather than interpolated
(a linear interpolator computes `sum(w_i v_i)` over the containing cell, and
`0 * NaN = NaN`, so a zero-weighted no-data neighbour voids a node that has a
perfectly good depth - it wrongly killed 792 cells around the formation's
edge); and MRST's COORD is written **column-major**, so a C-order ravel
produces a grid that builds happily, with finite coordinates throughout, and
is quietly scrambled - faces a hundred times too large, a tenth of the cells
collapsed to zero volume.

### Two findings about MRST 2024a

Cross-validation turned up two places where MRST itself is at fault, both
documented in the validators that found them:

- `ad-micp`'s solver reports convergence while its microbe equation still
  carries a residual of 1.8e-8; evaluating MRST's own residual at PyMRST's
  converged state gives 7.6e-15, so the 1.8% difference between the two is
  MRST's under-convergence, not ours.
- `pollock.m` tags each streamline point with the cell the line is about to
  *enter* rather than the cell its segment lies in (it stores `pos(:,1)`
  after advancing it). Of the 4050 points in the Phase-14 reference, only
  978 lie inside the cell MRST labels them with; all 4050 lie inside ours.

- `LimAzizShapeFactor` divides the matrix permeability back out of the shape
  factor (`sigma = pi^2 / (kx ky kz)^(1/3) * sum(k_i / l_i^2)`), while every
  transfer function in the dual-porosity module assumes sigma carries it -
  `KazemiTwoPhaseTransferFunction` even reads `km = rock_matrix.perm(:,1)`
  and then never uses it. Pairing them, as MRST's own API invites, inflates
  the transfer by a factor of 1/k: about 1e12 for a darcy matrix. PyMRST's
  `LimAzizShapeFactor` keeps the permeability in; pass `mrst_compatible=True`
  to reproduce MRST's number.

Two more are convergence, not correctness, but matter when comparing:
`standaloneFlash` leaves a fugacity error of ~2e-2 at its default tolerance,
and the dual-porosity models stop at `toleranceCNV = 1e-3` (a ~1e-5 error).
The Phase-13 and Phase-15 references tighten both.

### Note on MatlabBGL

MRST's cell-based `trapAnalysis` depends on MatlabBGL, whose downloaded
distribution ships no MEX binaries for Apple Silicon. To run the Phase-10
reference, `dfs.m` and `bfs.m` in `utils/3rdparty/matlab_bgl/matlab_bgl/`
were replaced with pure-MATLAB equivalents built on MATLAB's own `digraph`
(originals kept as `*.bgl-backup`). PyMRST itself has no such dependency.

## License

GPL-3.0-or-later (derived from MRST, © SINTEF Digital).
