# PyMRST: roadmap for the rest of the port

`README.md` records what is **built** (Phases 0-23). `GAPS.md` records what is
**missing**. This file says **how the missing part gets built**: in what order,
why that order, what each phase actually contains, and how each one is proved
correct.

It covers everything named in `GAPS.md` - about 50 unported modules plus the
holes inside the core - grouped into 14 phases. It deliberately excludes the
remaining TEDFM fracture workflows and the CMG-export pipeline, which are
tracked separately in `README.md`. The TEDFM VEM wrapper kernels themselves
landed in Phase 69.

## How to read this

Every phase below has the same five headings, because every phase has the same
five questions:

- **What** - the MATLAB files that get ported, by name.
- **Why here** - what it unblocks, and why it is not later.
- **Depends on** - phases that must land first.
- **Validation** - the specific thing that proves it. This is not optional; see
  the contract below.
- **Size / risk** - honest estimate, and where the estimate could be wrong.

Sizes are `S` (a session), `M` (a few), `L` (a substantial piece of work), `XL`
(a project in its own right). They are calibrated against completed phases:
Phase 23 (unstructured grids, ~1,200 lines, 5 validation cases) was an `M`.

## The validation contract

Nothing is "done" until it is checked against something that can disagree with
it. Twenty-three phases in, the pattern is settled and every new phase inherits
it:

1. **A MATLAB reference.** `dump_phaseN_reference.m` runs in the MRST tree next
   door and saves `.mat` arrays; `validate_phaseN.py` rebuilds the same case in
   PyMRST and compares field by field. Tolerances have run 1e-13 to 1e-16.
2. **Fix the inputs, compare the algorithm.** Where MATLAB and Python would
   differ on something upstream of the code under test - a triangulator, an RNG,
   a linear-solver tolerance - the *input* is dumped from MATLAB and fed to both.
   Phase 23 did this with triangulations; Phase 22 did it with random draws.
3. **Physics where MATLAB cannot be the judge.** Where the MATLAB path is
   unreachable or absent (the adjoint, `ve_dissolution`), the check is an
   invariant that would break if the code were wrong: a finite-difference
   gradient, a conservation law, an analytic solution. State which, and why.
4. **Self-tests in the file.** Every module runs its own unit tests under
   `python pymrst/.../module.py`, and `tests/test_all.py` collects them.
5. **Report what the reference gets wrong.** Five MRST defects have surfaced
   this way so far. "Matches MATLAB" and "correct" are not the same claim, and
   where they diverge the divergence goes in `README.md`.

## Active continuation rule

The compositional port is now being driven as an autonomous parity sprint. Work
continues phase-by-phase without asking for approval between ordinary
implementation steps. The only reasons to stop are real blockers: an ambiguous
scientific choice, a destructive filesystem action, an external install, a
MATLAB/system command that needs permission, or a validation failure that cannot
be resolved from local context.

Current compositional priorities, in order:

1. **Exact example parity.** Keep every MATLAB example under
   `mrst-2024a-Geomech/modules/compositional/examples` mirrored by a Python file
   with the same setup structure, not a wrapper.
2. **Facility-tree parity.** Extend the native facility model from AD-coupled
   separator groups into richer MRST-style group controls: shared separators,
   group limits, separator-controlled well equations, and control switching.
3. **Diagonal backend performance.** Keep replacing sparse fallbacks inside the
   diagonal backend with compact row-major/triplet operations. Direct row
   selection, row repeat, vertical stacking, row scaling, and CSR
   `matmul_left` are implemented, and Jacobian addition now stays in compact
   storage until solver handoff. Row-major addition is also Numba-backed when
   the optional compiled kernel layer is available; remaining work is deeper
   equation-assembly fusion and larger benchmark comparisons.
4. **Boundary-condition schedule richness.** Extend compositional BC controls
   across timesteps, including changing component streams and pressure/flux
   combinations.
5. **Broader MATLAB regression suite.** Each exact example port should gain a
   validation folder entry: MATLAB dump script, PyMRST validator, tolerance
   envelope, and pytest hook.

Current compositional sprint checkpoints:

- Phase 120 added a MATLAB-backed regression for MRST's
  `setupWellControlEquationsSingleWell` fallback behavior: BHP, total-rate,
  phase-rate controls, zero non-BHP targets, and missing oil/water/liquid
  streams now compare directly against MATLAB.
- Phase 121 added a MATLAB-backed regression for MRST's
  `TransportOverallCompositionModel` no-water/no-well transport step. PyMRST
  now stores the sequential pressure total flux on the state and uses
  density-weighted component transport fluxes, matching MRST's frozen-flux
  transport path to `8.3e-06` in `sT` and `1.3e-07` in composition.
- Phase 122 extended that transport parity to the water-active no-well path.
  The water transport equation now matches MRST's `rhoW*sW*sT`
  accumulation, `rhoW*sT` flux component, and `(1-sW)` oil/gas saturation
  scaling, comparing to MATLAB at `6.2e-06` in `sT`, `1.5e-07` in
  composition, and `3.8e-06` in phase saturations.
- Phase 123 extended sequential transport parity to the no-water well case.
  `pressure_step()` now stores MRST-style frozen pressure-step well streams
  (`well_flux` as reservoir phase flux and `well_components` as component
  mass flux), and `transport_step()` consumes those streams without solving
  `qOs`/`qGs`/`bhp` again. The new MATLAB regression matches transport
  `sT` to `6.4e-06`, composition to `6.4e-07`, phase saturations to
  `4.2e-06`, and well rates to `1.2e-09` m3/s.
- Phase 124 extended the same sequential well-transport parity to the
  water-active path. PyMRST now mirrors MRST's transport-iteration update of
  `wellSol.components`: each transport assembly stores the current component
  source back onto the state for the next Newton iteration. The MATLAB
  transport regression matches `sT` to `5.2e-06`, primary `sW` to `2.0e-06`,
  composition to `1.3e-07`, phase saturations to `3.3e-06`, and W/O/G well
  rates to `1.4e-10` m3/s.
- Phase 125 tightened the MRST-style schedule wrapper around sequential
  compositional solves. `run_schedule` now executes each report-step well
  control from an isolated copy, so control switches inside a Newton step do
  not mutate the stored `schedule.control` blocks or leak back into the model
  after the run. The new native regression drives
  `set_time_discretization('explicit')` through a two-control schedule and
  verifies model/schedule restoration.
- Phase 225 closed the core `params/wells_and_bc/psideh` gap used by several
  boundary-driven MRST examples. PyMRST now ports MRST's hydrostatic side
  pressure formula directly: boundary faces are selected by side alias and
  logical ranges, a mobility-weighted density is evaluated from
  `fluid.properties`/`fluid.relperm`, and `add_bc` receives one pressure value
  per face. The validator checks the exact formula, saturation broadcasting,
  side aliases, perpendicular `range` restriction, mapping-style fluid fallback,
  and error paths.
- Phase 226 threads that hydrostatic boundary helper into the compositional
  schedule-control parser. Boundary and mixed forcing controls now accept
  `psideh`/`pSideH`/`HydrostaticSide`, pass through MRST-style options such as
  `fluid`, `range`, `refPressure`, `refDepth`, `refPosition`, `gravity`, and
  `pn`, and consume those fields from `ScheduleControl.extra` after conversion.
- Phase 227 adds a direct MATLAB-backed regression for MRST's core `psideh`
  helper. `dump_phase227_reference.m` runs MRST on a small Cartesian grid and
  saves the selected hydrostatic boundary face, centroid, saturation,
  mobility-weighted density, gravity vector, and pressure value; PyMRST rebuilds
  the case and matches every field exactly.
- Phase 228 tightens the exact port of `compositionalBoundaryConditionsExample.m`.
  The PyMRST example now returns MRST-shaped plotting sections for pressure, gas
  saturation, and CO2 mole fraction for both natural variables and overall
  composition, plus the nonlinear-iteration comparison table plotted at the end
  of the MATLAB script.
- Phase 229 tightens the exact port of `compositionalExample3DSixComponents.m`.
  The PyMRST example now exposes the full MRST plotting loop as per-report
  frames: all six component fields, pressure, oil saturation, gas saturation,
  view metadata, subplot layout metadata, and cumulative report times, while
  keeping the legacy final-state field helper intact.
- Phase 230 tightens `book-ii/introToVaporLiquidEquilibrium.m`. The PyMRST
  VLE intro now exposes MRST-shaped SPE5 flash diagnostics in addition to the
  phase diagrams: an SSI iteration map, active-cell convergence history, and
  stability/flash timing table. The later MRST SSI+Newton/MEX branch is carried
  as an explicit unavailable slot, not fabricated data.
- Phase 231 tightens `book-ii/compositionalPerformanceExample.m`. The PyMRST
  benchmark still keeps its extra pure-Python `Diagonal (RowMajor)` backend row
  for implementation diagnostics, but the example now also exposes the exact
  four-row MRST timing table used for the plotted bar chart: Sparse, Diagonal,
  Diagonal (MEX), and Diagonal (MEX+RowMajor), with unavailable compiled rows
  represented by `NaN` timings and explicit reasons.
- Phase 232 tightens `compositionalLargerProblemTutorial.m`. The larger-problem
  tutorial now returns MRST's final plotting sections explicitly: the timing bar
  chart payload with `figure(1 + useNatural)`, tick labels, legend metadata, and
  the assembly-time zoom section with the same `ylim([0, 1.2*max(...)])` rule as
  the MATLAB script.
- Phase 233 tightens `book-ii/showNaturalOverall2.m`. The second natural-vs-
  overall single-cell example now records per-formulation wall time in addition
  to iterations, and returns the MRST final figure payload: two subplots with
  `Natural`/`Overall` bars titled `Iterations` and `Wall time`.
- Phase 234 tightens `book-ii/showNaturalOverall.m`. The first natural-vs-
  overall example now returns the two MRST convergence plot payloads: component
  composition vs. pressure change and liquid saturation vs. pressure change,
  both over a flash-derived saturation contour grid with the same legend,
  labels, solution marker, and `maxChange` line-style branch as the MATLAB
  script. Because PyMRST's compact nonlinear solver does not expose every
  sub-iteration state, the plotted paths are the available initial-to-final
  paths rather than fabricated iteration histories.
- Phase 235 tightens `book-ii/compositionalAccuracyExample.m`. The accuracy
  example now packages the four MATLAB plotting sections explicitly - component
  profiles, gas saturation, fluid properties, and cumulative iterations - with
  MRST legend placement, axis labels, y-limits, subplot/dual-axis semantics, and
  cumulative-iteration title metadata layered over the already-validated arrays.
- Phase 236 tightens `book-ii/compositionalMiscibilityExample.m`. The miscible
  displacement example now packages MRST's plotting metadata for the gas-front
  animation - figure number, `summer` colormap, line width, axes labels, and
  per-pressure profiles - and carries the final natural/overall iteration bar
  section with an explicit unavailable reason when the optional comparison
  branch is not run.
- Phase 237 tightens `runCompositionalSPE1.m`. The SPE1 comparison now packages
  the interactive plotting metadata from the MATLAB script: three `plotToolbar`
  sections with MRST titles, `axis tight`, and `view(30,30)`, the
  `plotWellSols` dataset-name bundle, and the cumulative nonlinear-iteration
  `stairs` plot metadata layered over the existing well/rate/iteration arrays.
- Phase 238 tightens `utils/setupSimpleCompositionalExample.m`. The shared
  simple-compositional setup now returns the MRST-shaped schedule object in
  addition to state/model/wells, clears well limits, assigns the same injected
  component stream to all schedule wells as the MATLAB utility does, and carries
  explicit unavailable metadata for MRST's optional `comparison.mat` reference
  file instead of leaving that fourth output implicit.
- Phase 239 tightens `compositionalValidationSimple.m`. The validation-simple
  example now packages MRST's comparison plotting metadata: component-profile
  comparison against MRST/E300/AD-GPRS slots, detailed pressure/saturation
  comparison with unavailable E300 payload explicit, `plotToolbar(...,
  'plot1d', true)` metadata, and ternary diagram triangle/label metadata.
- Phase 240 tightens `examples/Contents.m`. The compositional example catalog
  now carries an exact immutable `Contents.m` entry list with MRST ordering and
  one-line descriptions, while keeping the expanded PyMRST catalog for book-II
  examples and utility ports separate.
- Phase 241 tightens `compositionalExample3DSixComponents.m`. The six-component
  example now exposes MRST's result-plot sections explicitly: figure-position
  offset, subplot layout, component titles and `caxis([0, 1])`, pressure/sO/sG
  tail plots, per-step `drawnow` animation frames, and the final
  `plotToolbar`/view/axis metadata.
- Phase 242 tightens `compositionalBoundaryConditionsExample.m`. The boundary
  example now packages MRST's full plotting surface: permeability figure
  metadata, Natural/Overall `plotToolbar` titles, 3-by-2 comparison animation
  styling (`axis tight`, hidden tick labels, gas/CO2 color limits), and the
  `-o` nonlinear-iteration plot labels.
- Phase 243 tightens `compositional/properties/Contents.m` and
  `properties/components/Contents.m`. The property catalogs now preserve MRST's
  exact ordering, including the `CompositionalFlowPropertyFunctions` listing
  that appears in `Contents.m` even though no same-named MRST 2024a file exists;
  that row is represented explicitly as `missing_in_mrst_tree`.
- Phase 244 tightens `properties/components/EquationOfStateComponent.m`. The
  PyMRST component object now exposes MRST's camelCase component APIs
  (`getPhaseComposition`, `getComponentDensity`, and
  `getPhaseComponentFractionInjection`) on top of the existing Pythonic methods,
  including model/getProps and injection-force calling conventions.
- Phase 245 tightens `ad-core/models/facilities/separators/Contents.m`. The
  separator facility package now has an explicit MRST catalog for
  `BasicSeparator`, `EOSSeparator`, and `SeparatorGroup`, with implementation
  metadata and import checks tied to the PyMRST separator modules.
- Phase 246 tightens separator method-call parity. `BasicSeparator`,
  `EOSSeparator`, and `SeparatorGroup` result objects remain structured
  dataclasses, but now unpack like MRST's multiple-output calls
  (`streams, fractions, densities = ...`) for mass streams, mole streams,
  surface rates, and separator-stream diagnostics.
- Phase 247 tightens shared separator facility observability. When several
  producers route through one `SeparatorGroup`, `CompositionalState` now carries
  `separator_group_diagnostics` with producer indices/names, per-producer
  component mass streams, the summed separator input stream, allocation shares,
  positive group surface totals, and signed oil/gas rates allocated back onto
  each producer.
- Phase 248 threads those allocation diagnostics into
  `book-ii/compositionalSeparatorExample.m`'s example surface. The new
  `evaluate_separator_example_allocation_diagnostics` helper evaluates the
  no-separator, multi-stage separator, and single-separator alternatives in the
  same MRST problem order, while the packed-problem setup carries a diagnostics
  slot without forcing the optional separator-coupled Newton run.
- Phase 249 threads separator allocation diagnostics into compositional schedule
  reports. `_make_schedule_report`, `run_steps`, and `run_schedule` now carry a
  deep-copied `SeparatorAllocationDiagnostics` payload whenever the accepted
  state exposes `separator_group_diagnostics`, without treating diagnostics as
  a separator-control switch.
- Phase 250 adds a compact `SeparatorAllocationSummary` beside the full report
  diagnostics. Schedule reports now expose producer indices/names, allocation
  shares, signed allocated oil/gas rates, positive total surface oil/gas rates,
  and total component mass stream in a stable report-friendly payload.
- Phase 251 tightens `SeparatorGroup` API parity with MRST's MATLAB class. The
  Python class now accepts the `phaseDestination` constructor alias and exposes
  `phaseDestination`, `topologicalOrder`, `surfaceSeparator`, and
  `getTopologicalSort` beside the existing Pythonic names.
- Phase 252 tightens separator constructor/property parity. `BasicSeparator()`
  and `EOSSeparator()` can now be constructed empty like MRST objects populated
  later by options, and `EOSSeparator` exposes/preserves the MRST `EOSModel`
  property with a Pythonic `eos_model` alias.
- Phase 253 makes `EOSSeparator` honor its own `EOSModel` when provided. The
  separator flash now extracts the mixture from `sep.EOSModel` before falling
  back to the incoming model/mixture argument, matching MRST's local-EOS
  override semantics for separator objects.
- Phase 254 tightens `EquationOfStateComponent` property-name parity. Component
  objects now expose MRST-style `componentIndex`, `surfacePhaseMassFractions`,
  `surfacePhaseDensityPure`, and `molarMass` aliases that stay synchronized with
  the Pythonic fields used by the implementation.
- Phase 255 tightens `CompositionalMixture` method/property parity. `Mixture`
  now exposes MRST-style `getNumberOfComponents`, `getBinaryInteraction`,
  `setBinaryInteraction`, and a setter-backed `acentricFactors` alias while
  preserving the existing Pythonic helper names used by the flash solver.
- Phase 256 tightens `CompositionalPropertyModel` parity. The MRST-layout
  `fluids/compositional_property_model.py` file now exports a native
  `CompositionalPropertyModel` class with `computeMolarDensity`,
  `computeDensity`, `computeViscosity`, `computePseudoCriticalPhaseProperties`,
  and synchronized `volumeShift`/`volume_shift` properties.
- Phase 257 tightens `EquationOfStateModel` object-level parity. The
  MRST-layout `models/eos/equation_of_state_model.py` file now exports a native
  `EquationOfStateModel` shell with `CompositionalMixture`, `PropertyModel`,
  `omegaA`/`omegaB`, `setType`, `shortname`, cubic coefficient/root helpers,
  and component-name/count methods.
- Phase 258 ports MRST's small compositional fraction utilities. The
  MRST-layout `utils` package now exposes `ensureMinimumFraction`,
  `getNonUnitMassFraction`, and `getNonUnitMoleFraction` with Pythonic aliases,
  preserving the matrix and cell-array style behavior used by natural-variable
  conversion code.
- Phase 259 ports `getComponentsTwoPhaseSimpleWater`. The MRST-layout `utils`
  package now builds the same per-component phase-mass cell-array layout for
  water-active and no-water compositional paths, including the appended
  immiscible-water component when `model.water` is true.
- Phase 260 ports `formatMassString`. The MRST-layout `utils` package now
  exposes the mass-balance display formatter used by
  `checkComponentMassBalance`, including MRST's unit order, pluralization,
  sign handling, and decimal-at-depth `limit` behavior.
- Phase 126 extended native compositional facility group controls with
  deterministic group-target allocation modes. A violated group limit can now
  distribute the target by current rate share, equal share, or explicit
  fractions, with fractions re-normalized over active wells. The native
  regression checks name-based group membership, inactive-well handling, and
  state-guess resets after the group switch.
- Phase 127 pushed those facility controls through the MRST-style schedule
  layer. `simple_schedule` now accepts `src`, `separator_group`, and
  `group_controls` for one-control schedules, and `run_schedule` deep-copies
  active facility blocks before each report step so control switching cannot
  mutate the stored schedule or leak back into the model after the run.
- Phase 128 fixed mixed facility-control schedules. If a schedule never
  mentions separator/group controls, the model's default facility setup stays
  active during the run; once a schedule does manage those fields, report-step
  controls with `None` now clear stale separator/group blocks instead of
  reusing the previous control's facility settings.
- Phase 129 tightened the exact port of
  `book-ii/compositionalSeparatorExample.m`. The PyMRST example now builds the
  same MRST-shaped packed problem trio - `Overall-NoSep`,
  `Overall-MultiSep`, and `Overall-SingleSep` - over one scaled SPE5 depletion
  schedule, with the multi-stage and single separator groups attached to the
  corresponding models. Phase 116 remains the MATLAB-backed numerical
  separator-tree validation.
- Phase 130 expanded the exact port of
  `book-ii/introToVaporLiquidEquilibrium.m`. The PyMRST example now exposes
  MRST-shaped data sections for two-component Rachford-Rice VLE, ternary
  Peng-Robinson liquid-fraction maps, and scaled SPE5 pressure/temperature
  flash maps, while retaining the original simple pressure-line flash as a
  compatibility output.
- Phase 131 tightened the exact ports of `book-ii/showNaturalOverall.m` and
  `book-ii/showNaturalOverall2.m`. Both PyMRST examples now use MRST's
  single-cell grid, the `xmin` flux / `xmax` pressure boundary layout,
  explicit `bc.components`, and the MATLAB timestep defaults instead of the
  older two-cell boundary workaround.
- Phase 132 expanded `compositionalExample3DSixComponents.m` from a thin
  initialization smoke test into an MRST-shaped 3D quarter five-spot port:
  `lumped_1` six-component mixture, corner vertical wells, natural-variable
  model by default, pure-CO2 component streams, the two-year ramped schedule,
  plot-field extraction, scaled defaults, and a full-size `20 x 20 x 2`
  option.
- Phase 133 tightened `compositionalValidationSimple.m` around MRST's packed
  problem workflow. The PyMRST port now exposes `simple_comp` packed problems,
  `getMultiplePackedSimulatorOutputs`-style state/report/name outputs,
  explicit E300 and AD-GPRS comparison slots, and the ternary displacement
  paths used by the MATLAB plotting section.
- Phase 134 tightened `compositionalLargerProblemTutorial.m` around MRST's
  backend timing workflow. The PyMRST port now returns Sparse/Diagonal/MEX-row
  backend rows plus the Sparse+Direct solver row, separates assembly, linear
  solver, and total time, and exposes the same tick-label branch used by the
  MATLAB timing plot.
- Phase 135 tightened `book-ii/compositionalAccuracyExample.m` around MRST's
  plotting workflow. The PyMRST port now exposes component-profile frames,
  gas-saturation frames, viscosity/density/property overlays, and the
  cumulative implicit-iteration table in the same sections used by the
  MATLAB example.
- Phase 136 tightened `book-ii/compositionalMiscibilityExample.m` around
  MRST's result extraction sections. The PyMRST port now returns the
  pressure-sweep gas-saturation front animation data, raw packed
  states/reports, and the natural-vs-overall coarse timestep iteration table.
- Phase 137 tightened `runCompositionalSPE1.m` around MRST's plotting
  workflow. The PyMRST port now exposes the three well-solution datasets,
  reference `PseudoOil`/`PseudoGas` component-rate fields, and cumulative
  nonlinear-iteration stairs data.
- Phase 138 tightened compositional facility-tree controls. Producer/injector
  group limits can now reference direct wells, subgroup names,
  `group:<name>` entries, or nested `FacilityGroupControl` objects, with parent
  direct-member fractions pushed down to active leaf wells.
- Phase 139 added deck/schedule facility-tree import helpers for
  compositional controls. MRST-shaped `GRUPTREE`, `WELSPECS`, `WGRUPCON`,
  `GCONPROD`, and `GCONINJE` records now convert into native
  `FacilityGroupControl` trees.
- Phase 140 tightened compositional schedule reporting. `run_schedule`,
  `run_steps`, and packed-problem output helpers now expose MRST-style report
  dictionaries with one-based controls, control-change flags, nonlinear
  iterations, timestep cuts, accepted ministeps, and convergence flags.
- Phase 141 added deck/schedule well-control import helpers. MRST-shaped
  `WCONPROD`, `WCONINJE`, `WELTARG`, and `WELLTARG` records now update
  existing native `Well` objects with status, sign, controls, targets, limits,
  and injector phase streams.
- Phase 142 preserved deck/schedule group economics and network metadata.
  MRST-shaped `GECON` and `GRUPNET` records now land on imported
  `FacilityGroupControl.extra` dictionaries, including metadata-only child
  groups, so exact ports keep those rows until full enforcement is added.
- Phase 143 added MRST well-control keyword aliases to the compositional deck
  updater. `WCONHIST`, `WCONINJH`, and legacy `WCONINJ` now flow through the
  same native well-control update path as `WCONPROD`/`WCONINJE`, including
  `WELTARG`/`WELLTARG` target mutation.
- Phase 144 added deck-control schedule conversion for compositional examples.
  MRST-like `schedule.control` dictionaries can now become native
  `ScheduleControl` objects through `schedule_controls_from_deck_controls` and
  `simple_schedule_from_deck_controls`, applying deck well controls, importing
  group metadata, and preserving `bc`, `src`, separator, and group blocks for
  report-step switching.
- Phase 145 added conservative enforcement for imported `GECON` economics
  metadata. Producer groups now shut in when finite minimum oil/gas production
  or maximum water-cut/GOR/WGR criteria are violated, while ordinary group-rate
  limit switching still follows the existing allocation path.
- Phase 146 added a first native use of imported `GRUPNET` network metadata.
  Child-group guide rates with network use enabled now seed parent-group
  allocation fractions when no `WGRUPCON` fractions are present; the richer
  VFP/network behavior remains metadata for later facility-tree parity work.
- Phase 147 added boundary-control schedule conversion for compositional
  examples. MRST-like side-based and explicit-face boundary records now build
  native `BoundaryCondition` controls with changing flux/pressure values,
  saturations, and component streams for report-step switching.
- Phase 148 added source-control schedule conversion for compositional examples.
  MRST-like source dictionaries now build native `Source` controls with changing
  cell rates, saturations, and component streams for report-step switching.
- Phase 149 added mixed forcing-control schedule conversion. A single
  MRST-like report-step control can now carry both boundary records and
  source/sink records, and both are converted into native schedule blocks.
- Phase 150 added a structured compositional example-port catalog. The catalog
  maps every MRST `.m` example under `modules/compositional/examples` to its
  PyMRST module/file and validates that the mirror stays complete.
- Phase 151 added facility-switch diagnostics to compositional reports. Group
  rate-limit switches, GECON shut-ins, and well-control switches are recorded
  on the model and copied into MRST-style report dictionaries.
- Phase 152 added WGRUPCON status/metadata parity. Imported guide-control rows
  are preserved under `FacilityGroupControl.extra["WGRUPCON"]`, inactive rows
  no longer seed group-allocation fractions, and all-inactive guides fall back
  to rate/current allocation.
- Phase 153 added GECON end-run request reporting. Economics violations from
  imported `GECON` rows with `end_run=YES` now set model diagnostics, surface
  `EndRunRequested`/`EndRunGroups` in compositional schedule reports, and stop
  `run_schedule` after the accepted report step by default.
- Phase 154 added WGRUPCON fifth-column multiplier parity. The MRST
  `readWGrupCon` numeric multiplier field is preserved in
  `FacilityGroupControl.extra["WGRUPCON"]`, contributes to each row's
  `effective_guide_rate`, and drives native group allocation fractions.
- Phase 155 added MRST-style composition aliases for switched forcing
  controls. Boundary/source schedule dictionaries now accept `compi`, `zi`,
  `composition_i`, `comp_i`, and related aliases in addition to `components`
  and `z`, making exact ports less dependent on PyMRST-native field names.
- Phase 156 added MRST-style table boundary records for switched forcing
  controls. Boundary-control dictionaries can now carry parallel `face`,
  `type`, `value`, `sat`, and `compi`/composition rows in one record, including
  mixed `flux`/`pressure` entries, matching the shape of MRST `bc` structs more
  closely.
- Phase 157 added `GCONPROD`/`GCONINJE` metadata preservation. Imported group
  controls still expose native limits/status, but now also retain raw rows,
  active control mode, guide-rate fields, exceed/procedure flags, and late
  target columns in `FacilityGroupControl.extra` for future network/economics
  parity work.
- Phase 158 corrected `GCONPROD` guide-rate interpretation. Positional MRST
  rows no longer treat column 9 (`guide_rate` in `readGConProd`) as a native
  total-rate limit; the value is preserved as metadata, while explicit mapping
  aliases such as `target`/`total_rate` can still request a native `rate` limit.
- Phase 159 added `GCONINJE` duplicate replacement parity. Imported injector
  group controls now mirror MRST `readGConInje`: rows with the same
  `(group, phase)` replace earlier rows before native limits and metadata are
  formed, while different injection phases for the same group are preserved.
- Phase 160 canonicalized mapping-style `GCONPROD`/`GCONINJE` metadata raw
  rows. Sequence records remain untouched, but dictionary records now store
  `extra["GCONPROD"][i]["raw"]` and `extra["GCONINJE"][i]["raw"]` in the same
  fixed column order as MRST's `readGConProd`/`readGConInje` templates.
- Phase 161 extended canonical raw metadata ordering to mapping-style `GECON`,
  `GRUPNET`, and `WGRUPCON` rows. This keeps exact Python ports independent of
  dictionary insertion order while preserving sequence rows exactly as supplied.
- Phase 162 added `GECON`/`GRUPNET` duplicate replacement parity. Imported
  economics and network rows now mirror MRST `readGEcon`/`readGrupNet` by
  replacing earlier rows with the same group name before metadata is attached.
- Phase 163 added MRST-style well-control table replacement. Imported
  `WCONPROD`/`WCONHIST`/`WCONINJE`/`WCONINJ`/`WCONINJH` rows now keep only the
  last row for a well inside one keyword table and remove that well from other
  well-control keyword tables according to Python/deck keyword order before
  `WELTARG` is applied.
- Phase 164 added whole-well `WELOPEN` status parity. Imported deck controls
  now update the active post-replacement `WCON*` table for matching wells before
  `WELTARG` and native `Well` construction, while completion-level `WELOPEN`
  rows are conservatively ignored until PyMRST has a completion-control table.
- Phase 165 added MRST-style completion-status support. Native `Well` objects
  now carry optional `cstatus` flags, compositional well fluxes mask inactive
  completions through active `WI`, and wildcard-I/J/K `WELOPEN` rows with
  perforation-number bounds can open/shut selected completions without changing
  the stored Peaceman indices.
- Phase 166 added grid-aware positive-I/J/K `WELOPEN` matching. When
  `apply_deck_well_controls(..., grid=G)` receives a Cartesian grid, deck
  one-based I/J/K values now select matching well perforations using
  `cells.indexMap` when present, while `0` remains the MRST wildcard.
- Phase 167 threaded grid-aware `WELOPEN` matching through schedule conversion.
  `schedule_controls_from_deck_controls(..., grid=G)` and
  `simple_schedule_from_deck_controls(..., grid=G)` now forward Cartesian grids
  to well-control conversion so deck-style schedules can carry completion
  status changes per control block.
- Phase 168 added native `WELSPECS`/`COMPDAT` well geometry import for
  compositional deck controls. `wells_from_deck_completion_records` expands
  one or more completion rows into real PyMRST `Well` objects, preserves
  completion status as `cstatus`, passes explicit WI/radius/skin/direction
  metadata to `add_well`, and retains raw `WELSPECS`/`COMPDAT` rows for exact
  example ports.
- Phase 169 threaded that geometry import through deck-style schedule
  conversion. `schedule_controls_from_deck_controls(None, ..., grid=G, rock=R)`
  and `simple_schedule_from_deck_controls(None, ..., grid=G, rock=R)` can now
  bootstrap base wells from the first `WELSPECS`/`COMPDAT` control block before
  applying per-control `WCON*`, `WELTARG`, and `WELOPEN` updates.
- Phase 170 tightened COMPDAT completion-column parity. Defaulted connection
  transmissibility now falls back to the COMPDAT `KH` column through PyMRST's
  native Peaceman `compute_well_index(..., kh=...)` path, mixed explicit/default
  WI rows preserve explicit connection WI where supplied, and `SATNUM`, `KH`,
  and D-factor columns are stored per completion for exact ports.
- Phase 171 preserved MRST completion status text alongside boolean completion
  activity. Imported `COMPDAT` rows now store per-completion `cstatus_text`
  (`OPEN`, `SHUT`, or `STOP`) while `cstatus` remains the solver-facing boolean,
  and completion-level `WELOPEN` updates both representations on schedule-copy
  wells.
- Phase 172 ported MRST's default `COMPDAT` insertion for WELSPECS-only wells.
  `wells_from_deck_completion_records` now creates an all-layer vertical
  completion record for wells that appear in `WELSPECS` but have no explicit
  `COMPDAT`, using the WELSPECS I/J heel, open status, and MRST default
  0.1-diameter completion.
- Phase 173 matched MRST's duplicate/overlapping `COMPDAT` completion handling.
  Expanded completion rows are now collapsed by active cell with last-record
  wins semantics before `add_well`, so repeated perforations update WI, radius,
  status, KH/SATNUM/D-factor metadata, and ordering the way `processWells`
  does after expanding completion ranges.
- Phase 174 matched MRST's missing BHP defaults for imported deck controls.
  `WCONPROD` rows with absent BHP targets now default to one atmosphere, while
  `WCONINJE` rows default to MRST's high injector BHP value (`6895*barsa`);
  explicit infinite BHP targets remain unlimited.
- Phase 175 made legacy `WCONINJ` parsing MRST-column aware.  Full
  MRST-shaped `WCONINJ` rows now read BHP/THP from columns 9/10, while the
  newer `WCONINJE` rows continue to use columns 7/8 and older compact PyMRST
  analogue rows remain accepted.
- Phase 176 made `WCONHIST` BHP parsing history-control aware. Full
  MRST-shaped `WCONHIST` rows now read BHP from column 10, `WELTARG` updates
  that history BHP slot, compact native analogue rows remain accepted, and
  `WCONPROD` keeps its column-9 BHP / column-10 THP layout.
- Phase 177 ported MRST's historical `WCONHIST` `LRAT` target semantics. Full
  history rows now set the liquid-rate target from `-(ORAT + WRAT)` in columns
  4/5, instead of reading the ordinary `WCONPROD` LRAT column; compact native
  analogue rows and normal `WCONPROD` rows remain unchanged.
- Phase 178 ported MRST's historical `WCONINJH` rate-control semantics. Full
  history injection rows now have no control-mode column: column 4 is the
  active rate, BHP defaults to `6895*barsa`, `WELTARG RATE` updates the history
  rate slot, and older textual native analogue rows remain accepted.
- Phase 179 ported MRST's historical `WCONHIST RESV` semantics. Full history
  producer rows now import as `resv_history`, take the target from the negative
  sum of historical ORAT/WRAT/GRAT columns, and store MRST's WOG phase split
  instead of collapsing the control to ordinary `vrat`.
- Phase 180 ported MRST's historical `WCONHIST` phase-composition vectors for
  `ORAT`, `WRAT`, `GRAT`, `LRAT`, and `BHP`. Full history rows now update the
  producer stream composition just like `process_wconhist`, while compact
  native analogue rows keep their previous compatibility behavior.
- Phase 181 ported MRST's ordinary `WCONPROD` phase-composition vectors for
  `ORAT`, `WRAT`, `GRAT`, `LRAT`, `RESV`, and `BHP`. Producer controls now
  reset `w.compi` to the same WOG streams that MRST assigns in
  `process_wconprod`, with PyMRST's existing oil/gas projection for
  two-component compositional wells.
- Phase 182 preserves MRST `vfp_index` well metadata from deck controls and
  MATLAB well structs. `WCONPROD` and legacy `WCONINJ` read column 11, while
  `WCONINJE` reads column 9, matching `processWells.m`; the imported index is
  stored on `Well.vfp_index` for later binding to native `VFPTable` objects.
- Phase 183 binds positive `Well.vfp_index` values to native VFP tables during
  facility/model setup. Generic `FacilityModel` and native compositional models
  now select injector or producer tables by MRST's one-based index, and
  compositional `thp`/`vfp` controls close against the interpolated table BHP.
- Phase 184 converts MRST-like `VFPINJ`/`VFPPROD` schedule-control records into
  native `VFPTable` lists and binds them when schedules switch controls.
  Sparse one-based table numbers are preserved with empty placeholders, and
  empty selected slots now fail loudly instead of binding `None`.
- Phase 185 adds the MRST-layout facility selector
  `ad_core.models.facilities.select_facility_from_deck` /
  `selectFacilityFromDeck`, collecting deck `VFPINJ`/`VFPPROD` tables from all
  schedule controls and attaching them to native `FacilityModel` instances.
- Phase 186 exposes MRST-style intermediate separator stream diagnostics from
  `SeparatorGroup.compute_surface_mole_rates`: tree traversals can now return
  per-stage `molestreams`, `densities`, and `molefractions`, and
  `compute_surface_rates` carries those diagnostics on its result.
- Phase 187 wires those separator diagnostics into the MRST-layout
  `compositional_separator_example` output itself: the example now returns
  `tree_separator_streams` alongside `tree_surface_mole_streams`, and the
  packed-problem layout regression still checks the MRST problem ordering.
- Phase 188 adds `simple_schedule_from_mrst_schedule`, a direct converter for
  MRST-shaped schedule mappings with `step.val`, one-based `step.control`, and
  `control` blocks. It reuses the existing deck, boundary, source, VFP and
  facility converters, so exact example ports can pass the whole schedule object
  instead of manually splitting timesteps and controls.
- Phase 189 enforces shut-style `GCONPROD` exceed actions for producing groups:
  a violated group rate limit with `exceed_action` `SHUT`/`STOP` now shuts the
  active producer wells, zeros their state rate guesses, records the action in
  group-switch diagnostics, and leaves the existing redistribution behavior
  unchanged for `NONE`/default actions.
- Phase 190 adds `GCONINJE` injection-limit traceability: when an imported
  injector group limit is enforced, the corresponding metadata row records the
  last violation/action and the schedule report carries the MRST keyword,
  phase, and control mode that caused the redistribution.
- Phase 191 mirrors that traceability for ordinary `GCONPROD` redistribution:
  non-shut producer group switches now stamp the imported metadata row with the
  last violation, action, and exceed-action context while leaving shut-style
  `GCONPROD` handling unchanged.
- Phase 192 broadens exact MRST schedule ingestion: `simple_schedule_from_mrst_schedule`
  now accepts MATLAB/scipy-style struct objects and object-array controls in
  addition to dictionaries, preserving nested `BC`, `SRC`, deck/facility records,
  labels, and forcing restoration through the native PyMRST schedule runner.
- Phase 193 extends the same MATLAB/scipy-style struct support to the lower
  compositional facility deck converters themselves: WELSPECS/COMPDAT well
  import, WCON*/WELTARG well-control updates, GCON*/GECON/GRUPNET/WGRUPCON group
  conversion, and whole schedule facility conversion now accept object records
  directly instead of requiring pre-cleaned dictionaries.
- Phase 194 carries imported `GECON` economics context into runtime diagnostics:
  a conservative economics shut-in now stamps the metadata row with the last
  action and affected wells, and schedule reports include the MRST keyword,
  workover procedure, follow-on value, and end-run request that caused the
  shut-in.
- Phase 195 carries imported `GRUPNET` guide-allocation provenance into runtime
  diagnostics: parent groups whose fractions were seeded from child network
  guide rates now retain a `GRUPNET_ALLOCATION` metadata record, and group-limit
  switch reports keep `GCONPROD`/`GCONINJE` as the violated constraint source
  while reporting `GRUPNET` separately as the allocation source.
- Phase 196 promotes the preserved `GCONPROD` `NRAT`/network-rate column into a
  conservative native group limit. Imported positional and mapping rows now
  create `nrat` limits, producer groups redistribute against the total
  oil-plus-gas/water surface rate, and switched wells close with a matching
  total-rate `nrat` control equation.
- Phase 197 does the same for `GCONPROD` `RESV`/reservoir-rate rows. Finite
  positional or mapping `RESV` columns now import as native `vrat` group limits,
  reuse the existing total-rate well equation, and stamp the matching imported
  metadata row when redistribution occurs.
- Phase 198 lifts the late, mostly non-native `GCONPROD` columns out of raw-row
  storage into named metadata and runtime diagnostics. Damping,
  free-group-control, region, guide context, voidage/wet-gas/calorific/CRAT,
  TRGP, and procedure now survive in `extra["GCONPROD"]` and are carried into
  group-switch reports when the matching imported row is selected. These fields
  remain diagnostics unless a conservative native interpretation exists.
- Phase 199 mirrors that diagnostic traceability for `GCONINJE`. Surface and
  reservoir injection limits still use the conservative native redistribution
  path, while reinjection target, voidage target, guide rate/phase, TRGP, VREP,
  and wet-gas columns are now carried into the selected runtime report and the
  imported metadata row is stamped with the last violation/action.
- Phase 200 adds the same allocation-source traceability for `WGRUPCON`.
  Imported well guide rows that seed explicit group fractions now create a
  `WGRUPCON_ALLOCATION` metadata record, including active/inactive guide rows,
  guide multipliers, effective guide rates, and fractions. Runtime group-limit
  reports identify `WGRUPCON` as the allocation source separately from the
  violated `GCONPROD`/`GCONINJE` constraint and stamp the allocation metadata
  with the last violation/action.
- Phase 201 fills the native fallback in the same allocation-diagnostic path.
  Group limits that use PyMRST's built-in `rate`/`current`, `equal`, or
  explicit `fraction` allocation now report `NATIVE` allocation provenance with
  active wells, normalized allocation weights, and raw fractions where present.
  This does not change target distribution; it makes the native allocation
  decision inspectable in the same schedule reports as imported allocation
  provenance.
- Phase 202 makes schedule-level facility-block changes explicit in reports.
  `run_schedule` now emits `SeparatorChanged` and `GroupControlsChanged` flags
  when one-based control selection swaps managed separator/group blocks, and
  includes those control-block changes in `FacilityChanged`. Unmanaged
  schedules still preserve model-level facility defaults without reporting a
  facility change.
- Phase 203 extends those report diagnostics to managed well and VFP table
  blocks. Schedule reports now include `WellControlsChanged`,
  `VFPInjectorTablesChanged`, `VFPProducerTablesChanged`, and `VFPTablesChanged`
  flags when control selection swaps those facility inputs, and
  `FacilityChanged` includes those changes even if no nonlinear well/group
  switch occurred.
- Phase 204 adds the matching report visibility for schedule forcing blocks.
  `run_schedule` now emits `BoundaryChanged`, `SourceChanged`, and
  `ForcingChanged` when one-based control selection swaps managed `bc`/`src`
  inputs. These are deliberately separate from `FacilityChanged`, since
  boundary conditions and sources are external forcing rather than facility-tree
  controls.
- Phase 205 makes the simpler fixed-step runner use the same forcing-report
  dialect. `run_steps` now emits `BoundaryChanged`, `SourceChanged`, and
  `ForcingChanged`; a fixed `bc` is reported as a first-step forcing change and
  ordinary one-control report semantics remain unchanged.
- Phase 206 extends that fixed-step path to source forcing as well. `run_steps`
  now accepts `src`, forwards it to `model.step`, and reports fixed-source
  inputs through `SourceChanged`/`ForcingChanged` with the same first-step
  one-control semantics used for fixed boundary conditions.
- Phase 207 aligns fixed-step report schemas with scheduled reports. `run_steps`
  now emits false/empty defaults for facility-change flags, VFP table flags,
  well/group switch diagnostics, and end-run diagnostics, so consumers can read
  fixed-step and scheduled compositional reports through the same keys.
- Phase 208 carries the same report schema into packed-output fallbacks.
  `get_multiple_packed_simulator_outputs` now synthesizes full MRST-like
  reports when a legacy result only has the iteration vector, instead of
  returning bare `Iterations` dictionaries.
- Phase 209 makes those synthetic packed reports schedule-aware. When the
  packed problem still has `schedule.step.val/control`, the fallback report now
  recovers timesteps, singleton mini-step lists, one-based controls, and
  control-change flags; only mismatched legacy rows remain explicitly unknown.
- Phase 210 extends packed fallback reports to infer schedule block-change
  flags from the packed schedule controls. Synthetic reports now distinguish
  boundary/source forcing changes from facility changes and expose managed
  well, separator, group-control, and VFP table block changes when controls
  switch.
- Phase 211 broadens MRST-style boundary forcing control aliases. Boundary
  records now normalize `pres`/`p`/`bhp`/`dirichlet` to pressure controls and
  `rate`/`q`/`neumann`/`massrate` to flux controls, including side records,
  sequence records, face tables, and whole forcing-control metadata filtering.
- Phase 212 applies the same alias cleanup to source forcing controls. Source
  records now accept `q`, `massrate`, and `mass_rate` as rate fields in direct,
  nested, source-schedule, and mixed forcing-control records without treating
  source-only rows as boundary conditions.
- Phase 213 makes forcing-control metadata cleanup case-insensitive. Boundary,
  source, and mixed forcing schedule converters now consume parsed MATLAB-style
  aliases such as `Q`, `MassRate`, `COMPI`, and `ZI` instead of leaking them into
  `ScheduleControl.extra`.
- Phase 214 extends that case-insensitive handling to nested forcing container
  names. Mixed MATLAB/Python schedule dictionaries can now use
  `BoundaryConditions`, `BCRecords`, `SourceTerms`, or `SrcRecords` for nested
  boundary/source records, and those consumed container fields are filtered from
  schedule-control metadata.
- Phase 215 makes native forcing-object passthrough use the same lookup rules.
  Already-built PyMRST `BoundaryCondition` and `Source` objects can now sit under
  mixed-case aliases such as `Bc`, `BoundaryConditions`, `Src`, or `SourceTerms`
  in boundary, source, and mixed forcing controls, and are deep-copied before
  use.
- Phase 216 applies the same case-insensitive cleanup to schedule side facility
  blocks. MRST-style control dictionaries can now use `Wells`, `WELLS`,
  `SeparatorGroup`, `separatorGroup`, `GroupControls`, or `GROUPCONTROLS` in
  simple, MRST-shaped, boundary, source, and mixed forcing schedules without
  leaking consumed fields into `ScheduleControl.extra`.
- Phase 217 extends native forcing-object aliases into the generic and deck
  schedule paths. Already-built `BoundaryCondition`/`Source` objects under
  `BoundaryConditions`, `BCRecords`, `SourceTerms`, or `SrcRecords` are now
  deep-copied and consumed by `simple_schedule`,
  `schedule_controls_from_deck_controls`, and
  `simple_schedule_from_deck_controls`.
- Phase 218 makes the generic and deck schedule metadata filters
  case-insensitive too. Consumed uppercase aliases such as `WELLS`,
  `BOUNDARYCONDITIONS`, `SOURCETERMS`, `SEPARATORGROUP`, `GROUPCONTROLS`,
  `VFPINJ`, and `VFPPROD` no longer leak into `ScheduleControl.extra` after
  they have been parsed.
- Phase 219 broadens forcing container aliases for exact example ports.
  Boundary/source controls now accept singular and alternate names such as
  `Boundary`, `BoundaryCondition`, `SourceTerm`, and `SrcTerms` in direct,
  schedule, mixed forcing, generic, and deck schedule conversion paths, with the
  consumed fields filtered from `ScheduleControl.extra`.
- Phase 220 handles MATLAB cell-array forcing records. Boundary/source schedule
  controls loaded from `.mat` files can now arrive as `1 x N` or `N x 1` object
  arrays of record structs; the shared parser flattens those arrays into
  separate records while preserving numeric table rows.
- Phase 221 broadens forcing record field aliases for exact ports and `.mat`
  ingestion. Boundary/source records now accept `values`/`vals`,
  `faceIndex`/`faceIndices`, `cellIndex`/`cellIndices`, `saturations`, and
  `S`/`s` aliases, and the consumed aliases are filtered from schedule
  metadata.
- Phase 222 adds boundary side-name aliases used by exact ports and MATLAB
  struct dumps. Boundary controls now accept `faceSide`, `face_side`,
  `sideName`, and `side_name` wherever `side` was accepted, and schedule
  metadata cleanup consumes those aliases.
- Phase 223 adds typed boundary side aliases. Boundary controls can now use
  `pressureSide`/`pressure_side` and `fluxSide`/`flux_side` to encode both the
  side and the boundary type in one MRST-style field, and conflicting explicit
  `type` values fail loudly.
- Phase 224 adds MRST helper-name parity for pressure-side boundary controls.
  Exact ports and MATLAB struct dumps can now use `pside`, `pSide`, or `PSide`
  as pressure-boundary side aliases, with schedule metadata cleanup consuming
  those fields.
- `compositional_validation_simple.py` now follows the MRST example structure
  more closely: `use_natural` switch, explicit schedule, state history,
  nonlinear iteration report, and side-by-side natural/overall runner.
- Compositional well and group limit switching now resets the affected
  well-state primary guesses after a control change, mirroring MRST's
  facility-model behavior where switched controls also update `wellSol`.
- Generic AD-core and native compositional well-control equations now include
  MRST's wrong-direction BHP guard: a BHP-controlled well whose current surface
  rate has the opposite sign from `W.sign` falls back to the total-rate
  equation instead of continuing on an inconsistent BHP residual.
- Generic AD-core `FacilityModel` now supports MRST-style `resv` and `volume`
  controls: `resv` closes on surface phase rates weighted by
  `ControlDensity`/`control_density`, and `volume` closes on current
  reservoir-volume perforation fluxes (`cq_vol`).
- Compositional example infrastructure now includes MRST-shaped packed-problem
  helpers (`pack_simulation_problem`, `simulate_packed_problem`,
  `get_multiple_packed_simulator_outputs`). The accuracy and miscibility
  examples now build named packed cases before extracting states, reports, and
  names, matching the structure of the MATLAB examples more closely.
- Latest checks: `validate_phase120.py`, `validate_phase121.py`,
  `validate_phase122.py`, `validate_phase123.py`, `validate_phase124.py`,
  `validate_phase125.py`, `validate_phase126.py`,
  `validate_phase127.py`, `validate_phase128.py`,
  `validate_phase129.py`,
  `validate_phase130.py`,
  `validate_phase131.py`,
  `validate_phase132.py`,
  `validate_phase133.py`,
  `validate_phase134.py`,
  `validate_phase135.py`,
  `validate_phase136.py`,
  `validate_phase137.py`,
  `validate_phase138.py`,
  `validate_phase139.py`,
  `validate_phase140.py`,
  `validate_phase141.py`,
  `validate_phase142.py`,
  `validate_phase143.py`,
  `validate_phase144.py`,
  `validate_phase145.py`,
  `validate_phase146.py`,
  `validate_phase147.py`,
  `validate_phase148.py`,
  `validate_phase149.py`,
  `validate_phase150.py`,
  `validate_phase151.py`,
  `validate_compositional_suite.py`,
  `pymrst/modules/compositional/examples/_shared.py`,
  `pymrst/modules/compositional/examples/book_ii/compositional_accuracy_example.py`,
  `pymrst/modules/compositional/examples/book_ii/compositional_miscibility_example.py`,
  `pymrst/modules/compositional/examples/compositional_example_3d_six_components.py`,
  `pymrst/modules/compositional/examples/compositional_larger_problem_tutorial.py`,
  `pymrst/modules/compositional/examples/compositional_validation_simple.py`,
  `pymrst/modules/compositional/model.py`,
  `pymrst/modules/ad_core/models/facilities/facility_model.py`, and
  `pytest -q tests -k compositional` all pass. The remaining warning is the
  known pytest cache write warning.

## The one architectural decision that shapes everything

**PyMRST's AD models are hand-rolled; MRST's are not.**

PyMRST's `ad_blackoil`, `geothermal`, `dual_porosity` and `ad_micp` models each
assemble their residual directly, in one function. MRST stopped doing that years
ago. Its models are assembled from a **state-function graph**: `StateFunction`,
`StateFunctionGrouping`, `FlowDiscretization`, `FlowPropertyFunctions`,
`PVTPropertyFunctions` - a dependency DAG of small property evaluators that the
solver walks, caching as it goes. That is what `ad-core`'s 345 files mostly are.

Every downstream physics module in MRST - `ad-eor`, `solvent`, `ad-scal`,
compositional flow, most of `geothermal` - is written *against that API*. They
are not standalone residual assemblers; they are sets of state functions plugged
into the graph.

So there is a fork, and Phase 24 is where it gets taken:

- **Port the graph (Phase 24).** Every later physics phase becomes "write the
  state functions", which is what the MATLAB source already is - a close port
  rather than a re-derivation. Pay once.
- **Skip it.** Every later physics phase becomes a bespoke residual assembler,
  re-deriving equations that MRST expresses declaratively. Pay every time,
  and each payment is larger and less faithful than the port would have been.

This roadmap takes the first branch. It is the single highest-leverage item
remaining and it is why `ad_core` - not the meshing, not the geomechanics -
comes first. The existing hand-rolled models keep working; the graph is added
alongside them, and they are migrated onto it opportunistically (Phase 24 ports
one of them as its own proof).

---

# Tier 1 - Finish the core

Everything in this tier is depended on by something else. It comes first for
that reason, not because it is glamorous.

## Phase 24 - `ad-core`: the state-function graph `XL` — **DONE**

**Landed.** The AD-core graph now lives in MRST-shaped owner files:
`statefunctions/state_function.py` (the protocol: `StateFunction`,
`StateFunctionGrouping`, `State`, the DAG with per-state memoisation and
cycle detection), `statefunctions/*property_functions.py` and
`flow_discretization.py` (the PVT / flow / flux evaluators, generic over
2-phase and 3-phase live oil via a `PhaseFluid` adapter),
`models/physical_model.py`, `models/reservoir_model.py`,
`models/facilities/facility_model.py` (one well model instead of one per
fluid), `simulators/nonlinear_solver.py` (relaxation, line search, timestep
cutting, convergence reports), and
`ad_blackoil/models/generic_black_oil_model.py` (`GenericBlackOilModel`, assembled entirely
from the graph). The old flat files remain compatibility aliases only.

**Proved.** The migration was the proof, as planned: oil-water through the
graph reproduces the hand-rolled model **bit for bit** (0.0 difference), and
SPE 1 - three-phase, live oil, variable switching and flash - to **1.6e-14**.
Both still match MATLAB at the Phase 5a/5b tolerances (2e-14, 2.4e-11
relative). `examples/validation/validate_phase24.py`.

**What it cost, and what it did not.** Scope held: ~1,500 lines against
MRST's 345 files, because only the protocol plus black-oil's evaluators were
ported. The remaining `ad-core` surface (composite models, timestep
selectors, aquifers, sequential-implicit flow-state builders) is listed in
`GAPS.md` and gets added by the phases that need it - which is how MRST grew
it too.

*Original plan, for the record:*

**What.** Not all 345 files. The load-bearing subset:

- `statefunctions/`: `StateFunction`, `StateFunctionGrouping`,
  `StateFunctionDependent`, `FlowDiscretization`, `FlowPropertyFunctions`,
  `PVTPropertyFunctions`, `FacilityFlowDiscretization`, and the `flowprops/`,
  `flux/`, `facility/` evaluator sets.
- `models/`: `PhysicalModel`, `ReservoirModel`, `WrapperModel`, and the
  `getEquations` / `getProps` / `validateModel` protocol.
- `solvers/`: `NonLinearSolver` proper - line search, relaxation, convergence
  reports, cutting and retry - replacing the fixed Newton loop in `schedule.py`.
- `timesteps/`: `IterationCountTimeStepSelector`, `SimpleTimeStepSelector`.
- `initialization/`: `initEclipseState`-style equilibration helpers.

**Why here.** See above. Also: it is the thing the roadmap has been calling done
while it was 2 files.

**Depends on.** Nothing. Phase 4's `ADI` is the substrate and already exists.

**Validation.** Two-stage, and the second stage is the real one:

1. Graph mechanics: build a grouping, check the DAG resolves in dependency
   order, check caching (each state function evaluates once per state), check
   `getProps` returns AD-transparent values.
2. **Migrate `ad_blackoil` onto the graph and re-run Phase 5b unchanged.** SPE 1
   already matches MATLAB to ~1e-10. If the graph-assembled model reproduces
   that number, the graph is right - and the migration doubles as the worked
   example every later phase copies.

**Size / risk.** `XL`, and the risk is scope. The temptation is to port all 345
files; the discipline is to port the protocol plus enough evaluators to carry
black-oil, and let later phases add evaluators as they need them. The other risk
is Pythonic drift: MRST's graph leans on MATLAB's class system and dynamic
dispatch, and a faithful port must not turn into a Python framework of its own
invention, or every later "close port" claim evaporates.

## Phase 25 - Core leftovers: grid editing, rock, solvers, utils `M` — **DONE**

**Landed so far** (each MATLAB-validated to machine precision, same contract as
Phase 23):
- `utils/faceFlux2cellVelocity`, `faceFlux2cellFlux`, `cellFlux2faceFlux` ->
  `pymrst/utils/flux_conversion.py` (Phase 364): the lowest-order per-cell
  velocity reconstruction and the half-face/face flux conversions, needed by
  `mimetic` and any velocity post-processing. `validate_phase364.py` matches MRST
  to **0.0** on a 4x3x2 grid with a deterministic face-flux field.
- `gridprocessing/processNNC` -> `pymrst/gridprocessing/process_nnc.py`
  (Phase 365): resolves an explicit `[i1 j1 k1 i2 j2 k2 trans]` NNC list to
  active-cell pairs, with `exclude_existing`; the prerequisite for the fracture
  tiers' matrix/fracture coupling. `validate_phase365.py` matches MRST's resolved
  pairs and transmissibilities exactly, keep-all and excludeExisting.
- `params/rock/grdecl2Rock` -> `pymrst/params/rock/grdecl2rock.py` (Phase 366):
  assembles a `Rock` from GRDECL keywords - isotropic (`PERMX` -> 1 col),
  diagonal (`PERMX/Y/Z` -> 3), full symmetric (`+PERMXY/XZ/YZ` -> 6, in
  `[xx,xy,xz,yy,yz,zz]` order), plus PORO/NTG/ROCKTYPE, `MULTX/Y/Z` multipliers
  and fault data; `actmap`/grid restricts to active cells. `validate_phase366.py`
  matches MRST across all three tensor layouts number for number. The front half
  of every deck-driven case. **MRST quirk noted:** the isotropic collapse
  compares column *indices*, not values - three separate equal `PERMX/Y/Z` stay
  a 3-column tensor; only a lone `PERMX` (one column reused) collapses.
- `gridprocessing/makeInternalBoundary` ->
  `pymrst/gridprocessing/make_internal_boundary.py` (Phase 367): cuts a grid
  along a set of internal faces, replacing each with two coincident boundary
  faces (one per original neighbour) - how a sealing fault or explicit fracture
  is put into a grid, and the other fracture-tier prerequisite. Composed from the
  Phase-23 `add_faces`/`remove_faces` primitives. `validate_phase367.py` matches
  MRST's resulting topology exactly on a 3x3x2 cut - face/cell/boundary counts
  and the full neighbour-pair multiset (renumbering-independent; MRST's 1-based
  cells shifted to 0-based for the compare). Tag-marking of the new fault faces
  is deferred (PyMRST `add_faces` carries no tag argument yet).
- `gridprocessing/removeInternalBoundary` ->
  `pymrst/gridprocessing/remove_internal_boundary.py` (Phase 368): the inverse -
  welds each coincident boundary-face pair back into one internal face (merging
  the pair's nodes and restoring the two cells as neighbours). Validated by the
  decisive invariant a MATLAB dump cannot better: `make` then `remove` is the
  identity - the round trip recovers the original face count and neighbour-pair
  multiset exactly (contract rule 3). Completes the internal-boundary pair.
- `gridprocessing/splitDisconnectedGrid` ->
  `pymrst/gridprocessing/split_disconnected_grid.py` (Phase 369): splits a grid
  into the connected components of its cell-adjacency graph (interior faces plus
  NNCs), largest first, so a simulator does not hand a singular system to the
  linear solver when cells removal / a sealing fault has disconnected the domain.
  Uses `scipy.csgraph.connected_components` in place of MRST's `dmperm` (same
  partition). Invariant-validated (cells conserved, components isolated, sorted
  by size, cartDims propagated) - the partition is a graph property both codes
  compute identically, so a MATLAB dump adds nothing.
- `gridprocessing/hexahedralGrid` -> `pymrst/gridprocessing/hexahedral_grid.py`
  (Phase 370): builds a 3D grid from hex cell-node connectivity, finding shared
  faces via MRST's canonical face-orientation + `sortrows` trick (no geometric
  tolerance) - the entry point for an externally-meshed hex grid. Composed from
  `empty_grid`/`add_faces`/`rlencode`/`rldecode`. `validate_phase370.py` matches
  MRST on a two-cube mesh: identical cell/face/node counts and neighbour-pair
  multiset. **API note:** PyMRST takes `(hexes, coords)` 0-based; MRST's
  `hexahedralGrid(P, H)` is points-first, 1-based. Face direction tags are kept
  on `g.hftag` (PyMRST `add_faces` takes no tag argument).
- `gridprocessing/tessellationGrid` ->
  `pymrst/gridprocessing/tessellation_grid.py` (Phase 371): the 2D counterpart -
  a grid from a polygon tessellation (arbitrary polygons, not just tris/quads),
  faces are the polygon edges, shared edges found by the same
  smaller-node-first + unique-rows trick. Accepts a list of polygons or a
  uniform `(nc, k)` array. `validate_phase371.py` matches MRST on a 2x2 quad
  tessellation: identical cell/face/node counts and neighbour-pair multiset.
- `gridprocessing/triangulateFaces` ->
  `pymrst/gridprocessing/triangulate_faces.py` (Phase 372): splits each
  polygonal face into simplices through a face-centroid node (3D: `j`-gon -> `j`
  triangles; 2D: edge -> two sub-edges), each inheriting the parent's neighbours -
  what a VEM/MPFA/triangle-render path runs first. `validate_phase372.py` matches
  MRST on a 2x2x1 grid: face/node counts, neighbour-pair multiset, and the added
  centroid-node coordinates to 0.0.
- `gridprocessing/findMatchingFaces` ->
  `pymrst/gridprocessing/find_matching_faces.py` (Phase 373): groups faces by
  their (sorted) node set and returns the coincident pairs - the step that finds
  a fault's matching faces or an internal boundary's two sides (asserts at most
  two share a set, as MRST does; pads the node-set key with -1 since 0 is a valid
  0-based node). `validate_phase373.py` matches MRST after a 3x3x2 cut: 17 pairs,
  matched-pair midpoints to 0.0.
- `gridprocessing/refineGrdecl` + `grdeclXYZ` ->
  `pymrst/gridprocessing/refine_grdecl.py` (Phase 374): subdivides a corner-point
  (COORD/ZCORN) deck by integer factors `(nx, ny, nz)`, interpolating the pillar
  geometry linearly and inheriting cell properties (PERMX/PORO/ACTNUM/...) from
  parent to child - the standard grid-resolution study on a real corner-point
  model. `grdecl_xyz` is the small COORD/ZCORN unpacker it needs.
  `validate_phase374.py` matches MRST on a wavy 3x2x2 deck refined [2 2 2]:
  cartDims, COORD, ZCORN and inherited PERMX all to 0.0. FAULTS refinement is
  deferred (needs the deck fault-record structure).
- `gridprocessing/get2DGridFromFaces` ->
  `pymrst/gridprocessing/get_2d_grid_from_faces.py` (Phase 375): turns a set of
  3D-grid faces into a 2D grid (each face -> a 2D cell, its edges -> 2D faces) -
  the surface/cross-section extractor `sliceGrid` builds on, and a fault-surface
  mesher on its own. `validate_phase375.py` matches MRST on the top faces of a
  3x3x2 grid: cell/face counts and neighbour-pair multiset. **MRST bug found:**
  MRST sets the 2D grid's `nodes.num` to the *edge* count, inconsistent with its
  own coords array; the port sets it to the true node count (16, matching MRST's
  actual 16 coord rows, not its buggy `nodes.num` of 24).
- `solvers/initWellSol` -> `pymrst/solvers/init_well_sol.py` (Phase 376): the
  blank per-well solution (zero perforation fluxes, starting pressure `p0`) the
  first solve writes into; the well-side companion to `initResSol`. Self-tested
  (trivial).
- `solvers/accumulateCartesianMultipliers` ->
  `pymrst/solvers/accumulate_cartesian_multipliers.py` (Phase 376): turns
  per-cell directional `MULT[X/Y/Z]` keywords into a per-face multiplier by
  matching each half-face's direction (from `cells.tags`, the 0-based form of
  MRST's `cells.faces(:,2)`) and taking the product on both sides. How
  directional deck multipliers reach the transmissibility. `validate_phase376.py`
  matches MRST on a 3x2x2 grid to 0.0.

- `gridprocessing/glue2DGrid` -> `pymrst/gridprocessing/glue_2d_grid.py`
  (Phase 377): the full 478-line stitcher - joins two 2D grids along a shared
  boundary by concatenating, tracing each boundary as an oriented loop, merging
  coinciding nodes within a tolerance, welding the boundary faces between the two
  grids, and *splitting* a coarse boundary edge where the other grid has an extra
  node (conformal insertion). `validate_phase377.py` matches MRST on both a
  node-matching case (two squares -> 2 cells, 7 faces, 6 nodes) and a
  non-conforming case that exercises the edge splitting (coarse square + two
  stacked squares -> 3 cells, 10 faces, 8 nodes) - cell/face/node counts and the
  neighbour-pair multiset all identical. One of the two big geometry pieces done.

- `gridprocessing/repairNormals` -> `pymrst/gridprocessing/repair_normals.py`
  (Phase 378): recomputes consistent 2D geometry - cell areas ("volumes"), face
  lengths ("areas") and outward-consistent normals - from node coordinates, for
  the 2D grids `tessellationGrid`/`glue2DGrid` produce with topology only.
  `validate_phase378.py` matches MRST on an irregular 2-cell grid: cell areas and
  sorted face lengths to 0.0.

- `gridprocessing/removeShortEdges` ->
  `pymrst/gridprocessing/remove_short_edges.py` (Phase 379): collapses edges
  shorter than a tolerance (slivers from corner-point processing / gluing),
  merging their endpoints and cleaning up the collapsed faces and cells -
  composing `findMatchingFaces` + `removeInternalBoundary` + `removeCells`/
  `removeFaces` built alongside it (scipy `connected_components` for MRST's
  `dmperm`). `validate_phase379.py` matches MRST on a pentagon-with-sliver ->
  quad: counts and merged coordinates to 0.0.
- `gridprocessing/cellNodes` -> `pymrst/gridprocessing/cell_nodes.py`
  (Phase 380): each cell's corner nodes in canonical local-vertex order
  (WSD..ENU), found by summing a power-of-two per incident face direction into a
  binary corner key - using the half-face direction tags from `cells.tags` (0-5,
  the 0-based form of MRST's `cells.faces(:,2)`). `validate_phase380.py` matches
  MRST's full [cell, local-vertex, node] table on a 2x2x1 grid exactly.

- `gridprocessing/removePinch` -> `pymrst/gridprocessing/remove_pinch.py`
  (Phase 381): removes pinched-out layers - merges coincident nodes, drops
  collapsed faces/cells, and reconnects the faces that became coincident across
  the pinch (its `uniqueFaces` = `findMatchingFaces` + `removeInternalBoundary`).
  MRST's canonical face-node reordering is skipped: it exists only to let its
  `uniqueFaces` match by node *sequence*, whereas `findMatchingFaces` matches by
  node *set* (order-independent), so the reconnected topology is identical.
  `validate_phase381.py` matches MRST on two duplicated squares -> reconnected
  2-cell grid: counts and neighbour-pair multiset.

- `utils/gridtools/computeGridSlicePolygons` (core) ->
  `pymrst/gridprocessing/compute_grid_slice_polygons.py` (Phase 382): the
  geometric heart of `sliceGrid` - the per-cell cross-section polygons where a
  plane cuts a 3D grid. Defines the plane (point+normal or segment+cutDir), finds
  each face-edge crossing, groups the points by cell, and orders each cell's ring
  by angle in the 2D cut-plane frame. `validate_phase382.py` matches MRST on a
  vertical x=1.5 cut of a 3x3x2 grid: same cut cells and vertex counts, all
  intersection coordinates to 0.0. **Scoped:** the 585-line MATLAB routine's
  extras - bounding-box face pre-selection (speed only), node snapping,
  ellipsoid/radius clamping, finite-segment `clampSides`, and multi-segment
  polylines - are deferred (the validation uses an over-spanning segment so
  clamping is a no-op); they refine/accelerate the same core result.

- `gridprocessing/sliceGrid` (cell-splitting) ->
  `pymrst/gridprocessing/slice_grid.py` (Phase 383): cuts a 3D grid along a
  plane, splitting each crossed cell in two, each crossed face in two, and
  inserting the cross-section polygon as the internal cut face between the
  halves. Implemented with a robust global edge-crossing (marching)
  formulation - crossing nodes per straddling edge, polygon-clip of each cut
  face, and a Newell-oriented cut face so `compute_geometry` signs the halves
  correctly. `validate_phase383.py` matches MRST's `sliceGrid` on a 3x2x2 grid
  cut at x=1.5: cell count (16), total volume (12) and the sorted per-cell
  volumes to 0.0.
- **Node-snapping refinement (Phase 384):** the marching formulation handles a
  plane that passes exactly through grid nodes - on-plane nodes (`side == 0`) are
  shared cut-polygon vertices, no crossing node is made on their edges, and a face
  wholly on the plane is kept as-is. `validate_phase384.py` matches MRST on the
  diagonal x=y cut of a 2x2x1 grid (through the corner nodes): 6 cells, sorted
  volumes to 1e-16.
- **Partial-cut refinement (Phase 385):** a finite cutting segment now bounds the
  cut - each would-be-cut cell is checked against the segment's `[0, L]` window
  along its direction, and cells the segment only partially reaches are left whole
  (MRST's `removePartialCuts`, which `sliceGrid` always sets). A crossing face
  whose neither neighbour is fully cut is kept whole. `validate_phase385.py`
  matches MRST on a segment stopping at x=2.5 across a 3x1x1 grid: cells 0,1 cut,
  cell 2 whole (5 cells), sorted volumes to 0.0.
- **`sliceGrid` is now complete for all cut types** - clean interior, node-snapped,
  and partial - each MATLAB-validated to machine precision. Remaining out of scope:
  the ellipsoid/radius clamping and bbox speed options of `computeGridSlicePolygons`
  (refinements/optimisations of the same result), and a plane coinciding with a
  *cut* cell's whole face (the zero-volume degenerate MRST also warns about),
  which is refused cleanly.

- `params/rock/permeabilityConverter` ->
  `pymrst/params/rock/permeability_converter.py` (Phase 386): rotates a
  cell-local diagonal permeability (`PERMX/Y/Z` measured along a cell's own tilted
  geometry) into the full symmetric global tensor `sum_i k_i (e_i ⊗ e_i)`, the
  local axes `e_i` recovered from the corner geometry (`cell` /
  `bedding_plane` / `bedding_plane_normal` frames). `validate_phase386.py` matches
  MRST on a tilted corner-point deck: all six tensor components to machine
  precision (~1e-15). **MRST bug found & refused:** the off-diagonal *input*
  branch (`PERMXY/YZ/ZX` supplied) has its component-index `switch` inside a loop
  but the tensor update outside it, so only the last field's indices are used and
  with the wrong magnitude; the port refuses that input rather than reproduce it.

**Phase 25 is complete.** All grid-editing / rock / solver leftovers are ported
and MATLAB-validated (Phases 364-386), including the full `sliceGrid` (clean,
node-snapped and partial cuts). The MPFA and fracture tiers are fully unblocked
on the grid side.

**What.** The small named gaps, which are individually trivial and collectively
load-bearing:

- `gridprocessing/`: `sliceGrid`, `makeInternalBoundary` /
  `removeInternalBoundary`, `refineGrdecl`, `removePinch` / `processPINCH`,
  `processNNC`, `splitDisconnectedGrid`, `repairNormals`, `removeShortEdges`,
  `findMatchingFaces`, `triangulateFaces`, `cellNodes`, `hexahedralGrid`,
  `tessellationGrid`, `glue2DGrid`, `get2DGridFromFaces`.
- `params/`: `grdecl2Rock`, `permeabilityConverter`, `psideh`.
- `solvers/`: `accumulateCartesianMultipliers`, standalone `initWellSol`.
- `utils/`: `faceFlux2cellVelocity` / `cellFlux2faceFlux`.
  (`multiEig` / `multiSymmEig` landed with VEM stress utilities in Phase 68.)

**Why here.** `processNNC` and `makeInternalBoundary` are prerequisites for the
fracture phases (Tier 4). `faceFlux2cellVelocity` is needed by `mimetic` and by any velocity
post-processing. Doing them as one sweep is cheaper than doing them five times
as side-quests, and Phase 23 already built the primitives (`add_faces`,
`remove_faces`, `remove_cells`) they compose from.

**Depends on.** Phase 23 (done).

**Validation.** Direct MATLAB dumps, same shape as Phase 23. Each function gets
a case; grid editors are checked on a corner-point grid with faults (where
`repairNormals` and `findMatchingFaces` actually do something) rather than on a
Cartesian box where they are no-ops.

**Size / risk.** `M`. Low risk - these are small, well-specified functions. The
one to watch is `sliceGrid`, which is genuinely intricate polygon clipping.

## Phase 26 - `upr`: PEBI mesh generation `L`

**What.** `upr/pebi2D/` (well- and fault-conforming 2D PEBI: `pebiGrid2D`,
`compositePebiGrid2D`, the protection-site and clipping machinery),
`upr/util/`, and `upr/distmesh/` (the DistMesh iterative point placer). 3D
(`pebi3D`) is a stretch goal, not a commitment.

**Why here.** Phase 23 built `pebi`, which *consumes* a point set and produces a
Voronoi grid. Nothing in PyMRST *produces* a good point set - scipy's Delaunay
on an arbitrary cloud is not a reservoir mesh. `upr` is the missing half: it
places sites so that cell faces align with faults and cells align along
wellbores, which is the entire reason to use PEBI grids in reservoir simulation.
Without it, Phase 23 is a capability you cannot really use on a real field.

**Depends on.** Phase 23.

**Validation.** MATLAB dumps of `pebiGrid2D` and `compositePebiGrid2D` on the
`upr` examples' own well/fault geometries. Fix the point set where DistMesh's
iteration would diverge between the two languages (it is a relaxation loop with
a convergence tolerance - dump MATLAB's converged sites and feed them to the
Python `pebi` to compare grid construction; separately check the Python
DistMesh converges to sites within tolerance of MATLAB's).

**Size / risk.** `L`. The risk is DistMesh: it is an iterative optimizer, so
bit-exactness is not on the table and the validation has to be structured around
that (see above). Everything downstream of the sites - the clipping, the
conformity - should still match exactly.

---

# Tier 2 - Geomechanics

The reference tree is named `mrst-2024a-Geomech`. `vemmech` (VEM elasticity) and
linear Biot are already ported and validated. This tier is the rest of the
coupled-geomechanics story, and it is the most defensible "next capability" the
project has.

## Phase 27 - `ad-mechanics`: coupled flow and mechanics `L` — **DONE**

**Landed.** `mechanics.py` (`MechanicModel` - poroelastic mechanics on VEM,
with the fixed-stress terms), `poromech.py` (`MechWaterModel` fully coupled,
`FixedStressSplitModel` sequential), `mandel.py` (the benchmark and its
analytic solution). `vemmech` gained `vem_strain_operator`,
`vem_stress_operator` and `inv_ci`. Later VEM completion passes added the
standalone stress/principal-stress utilities and batched eigensolvers
(Phase 68), then the TEDFM VEM wrapper kernels on `G.Matrix` grids
(Phase 69), the `VEM_linElastTEDFM` solve wrapper (Phase 70), and the
`vemmech/grid_utils` helpers (Phase 71), followed by backend-agnostic
deformed plotting geometry helpers (Phase 72), and the first numeric AD VEM
tensor utility substrate (Phase 73). Phase 74 added the first calibration
helper slice: principal-stress conversion, boundary element identification,
and numeric material/load closures with K/G/rho layer controls. Phase 75
added the calibration boundary-condition helper (`set_bcfun`) with active
boundary controls and MRST's roller/z-lock displacement semantics. Phase 76
added the numeric value path for calibration `calculate_stress`. Phase 77
added the numeric Gaussian objective value/setup slice (`setup_objective_fun`
without AD displacement elimination or adjoint gradients). Phase 78 started
the full AD VEM path with numeric `VEM_assemble_AD`/`VEM_linElast_AD` surfaces,
AD-shaped sparse-tensor BC/load inputs, reduced solve, `no_solve`, and `disc`
operators.
Phase 79 added the first 3D material AD derivative layer: `compute_ax_derivs`
and `extra.Ax_derivs` for ADI Young's modulus/Poisson ratio inputs, verified
against central finite differences of the VEM stiffness action. Remaining AD
VEM work at that point was full lazy tensor parity, AD RHS/BC derivatives, and
hooking this into the calibration displacement-elimination objective. Later
calibration phases supplied the RHS/BC/material/load/objective derivative path
through explicit eager/analytic blocks; generic MRST lazy sparse-tensor parity
remains a separate substrate item.
Phase 80 hooked `setup_objective_fun.objfun` to an eliminated mechanics solve
and finite-difference control gradient, with `op_0` initialized through
`VEM_linElast_AD(..., no_solve=True)`.
Phase 81 generalized material `Ax_derivs` to 2D VEM grids, fixed the AD
assembly path to use the 2D elasticity tensor in 2D, and validated the 2D
`dAx/dE` and `dAx/dnu` columns against central finite differences.
Phase 82 replaced the full finite-difference eliminated-objective gradient
with the MRST adjoint structure (`oval_du + dsys_du' * lambda`) while keeping
the partial derivative blocks finite-differenced for now. It also added
`vem_reduced_system_residual`, validated at the solved displacement and
against the reduced stiffness matrix action.
Phase 83 made calibration material closures AD-aware for stiffness controls:
`set_material_fun(...).efun/nufun` preserve PyMRST `ADI` Jacobians for K/G
controls. Phase 84 uses those material AD columns in the calibration adjoint
through `compute_ax_derivs`, leaving finite differences only for the remaining
boundary/load/objective partial blocks. Phase 85 added analytic
boundary-condition derivative columns from `set_bcfun` and uses them for the
reduced mechanics residual columns in the calibration adjoint.
Phase 86 added load/density residual derivative columns for rho controls in
`set_material_fun.load_jac` and uses them in the calibration adjoint; the
remaining finite-difference derivative blocks at that point were the objective
partials `oval_du`/`oval_dd`. Phase 87 replaced the displacement-side objective
partial `oval_dd` with a stress-operator gradient
(`objective_displacement_gradient`), validated against the old free-DOF central
finite-difference path. Phase 88 replaced the whole-objective
finite-difference `oval_du` block with `objective_control_gradient`: analytic
prior, analytic boundary-control stress terms, and localized stress
sensitivity columns. Phase 89 replaced the local stress-observation
finite-difference bridge with analytic derivatives for principal stresses,
vertical stress, and orientation mismatch. Phase 90 added
`objfun_with_hessian`, an optional Hessian and Hessian-direction callback for
the eliminated objective, while keeping the default MRST-compatible `objfun`
return shape with empty Hessian slots. The Hessian path now propagates
boundary-control eliminated mechanics through ADI2 for constant-material
calibration cases.
Phase 91 added the first calibration workflow driver: `calibrate_model`,
`produce_syndata`, and `optimize_mech`, with synthetic-data mode, MRST-style
result fields, and validation on a one-cell synthetic calibration case.
Phase 92 added an SR1 trust-region optimizer slice patterned after MRST
`optimizeSR1`, including trust-radius adjustment and SR1 Hessian updates.
Phase 93 added backup-file continuation and progress callbacks for the SR1
workflow. Plotting is deliberately exposed through the callback hook rather
than a MATLAB-style hardwired figure.
Phase 94 added first-order ADI coefficient propagation through the eager
`SparseMultiArray` tensor substrate, `kron_adi`, `face_node_tensor`, and
`Enu2C_AD`, with finite-difference checks for the elasticity tensor
coefficients. At that point, the remaining parity gap was the compiled/lazy
part of MRST's generic tensor engine (`tbinaryop`/`tcontract`) and exact
second-order lazy AD Hessians, not the runnable calibration workflow.
Phase 95 added the pure-Python side of that substrate: `LazySparseTensor`,
`tbinaryop`, and `tcontract` record tensor expression graphs before
realization, and `ADI2`/`hessian_ad2` provide exact second-order Hessians for
scalar AD expressions. This gives PyMRST a runnable lazy tensor and Hessian
substrate; the remaining boundary is compiled MEX backend identity/performance
and byte-for-byte identity with MRST's compiled tensor kernels.
Phase 96 added material-control Hessians for the eliminated calibration
objective: K/G stiffness controls and rho load controls are included through
the validated eliminated-gradient path and checked against finite-difference
gradient references.

**The Phase-24 dividend, measured.** Poroelasticity turned out to be *one
state function*: the pore volume grows with the rock's dilation, so
`PoroMechanicalPoreVolume` is a drop-in replacement for `PoreVolume` and
nothing downstream changes. The fully-coupled and fixed-stress models differ
only in which pore volume they install. That is what the graph was for, and
it held.

**Proved.** Against MATLAB `ad-mechanics` (4e-9 pressure, 9e-10
displacement); against Terzaghi (0.3%); and against **Mandel** - the
Mandel-Cryer overshoot reproduced at peak 1.096 vs analytic 1.098, whole
profile under 1%, error falling under refinement. Fixed-stress splitting
contracts to the fully-coupled answer to 5e-7.
`examples/validation/validate_phase27.py`.
The Norne field comparison now also validates full displacement, volumetric
strain, full strain, effective stress and total stress directly against MRST's
VEM operators in `dump_norne_reference.m` / `validate_norne.py`.

**Multiphase, too.** `MechBlackOilModel` (MRST's `MechOilWaterModel` /
`MechBlackOilModel`) came in at ~150 lines: it is `GenericBlackOilModel` with
the same `PoroMechanicalPoreVolume` installed and the displacement added as a
primary variable. The multiphase physics needed no re-derivation at all -
the third and cheapest dividend from Phase 24.

**What it cost.** Two genuine bugs of our own, both about the *previous*
state's displacement, and both of which passed Terzaghi while silently
destroying Mandel - vindicating the decision to make Mandel the acceptance
test rather than a nice-to-have. Both are written up in `README.md`.

**And two MRST defects.** A sixth: `ad-mechanics` under-converges at its
default tolerances (2e-7 drift across a schedule). A seventh, and the most
consequential found so far: **`equationsOilWaterMech` omits `G.cells.volumes`
on its Biot term**, so the poromechanical coupling in MRST's oil-water model
is too weak by a factor of the cell volume - a 10% pressure error on 500 m3
cells, and dimensionally inconsistent, so its strength depends on the grid's
unit of length. `validate_phase27.py` pins it rather than asserting it: on a
500 m3 grid, restoring the missing factor reproduces MRST to **2.8e-15**,
while a correct term differs by 9.9e-2; on a 1 m3 grid the two formulas
coincide and both match. `MechWaterModel` and `MechBlackOilModel` have the
factor; only the oil-water equation is wrong.

**And the adjoint.** `adjoint.py` computes the gradient of **surface uplift**
(MRST's `objUplift`) with respect to every well control at every timestep in
one backward sweep - the question coupled geomechanics is actually asked:
*how should the wells run so the ground stays still?* Verified by finite
differences, as the flow adjoint was: the full gradient to 2.8e-10, and the
two blocks it adds (`dR/du`, `dR/dx_{n-1}`) each checked separately.

Porting it caught a latent trap in the Phase-24 graph:
`component_conservation_equations` wrapped the previous mass in `value()`,
which is harmless forwards but silently zeroed the whole `dR/dx_{n-1}` block
once the adjoint seeded that state as an AD variable. The hand-rolled models
warned about exactly this in a comment; the graph had lost it.

**Phase 27 is complete.** The only unported thing in `ad-mechanics` is the
Norne field case, and that is a *dataset* dependency
(`getDatasetPath('norne')`) rather than a capability - it runs the models
above on a GRDECL grid PyMRST already reads bit-exactly (Phase 6).

*Original plan, for the record:*

**What.** 44 files. `MechanicModel`, the fully-coupled models (`MechBlackOilModel`,
`MechOilWaterModel`, `MechWaterModel`) with their `MechFluidModel` /
`MechanicMechModel` helper hierarchy, and the **fixed-stress splitting**
(`MechFluidFixedStressSplitModel`) - the sequential scheme that makes coupled
poromechanics affordable at field scale.

**Why here.** It is the natural extension of the two things already validated
(`vemmech` elasticity, linear Biot), it is what the tree is named after, and it
comes with the best validation targets in the whole port.

**Depends on.** Phase 24 (these are AD-OO models), existing `vemmech`.

**Validation.** Unusually strong, on three levels:

1. **Analytic.** `ad-mechanics/examples/terzaghi.m` and `mandel.m` are the
   canonical poroelasticity benchmarks and both have closed-form solutions.
   Terzaghi is already matched to 0.2% by the Phase-7a Biot model; Mandel is
   harder (it has the non-monotonic pressure overshoot that catches wrong
   coupling) and is the real test.
2. **MATLAB.** Dumps from `run2DCase.m` and the Norne case.
3. **Physical.** Fixed-stress splitting must converge to the fully-coupled
   solution; that contraction is its own test, as sequential splitting's was in
   Phase 15.

**Size / risk.** `L`. Risk is the mechanics solver: MRST leans on `vemmech`'s
assembly and on direct solves that get expensive fast. Norne may need to be
validated on a subset.

## Phase 28 - `fvbiot` + `mpsaw`: finite-volume stress `L`

**What.** `fvbiot` (22 files: MPSA stress + MPFA flow + the Biot coupling terms,
`mpsa.m`, `mpfa.m`) and `mpsaw` (91 files: multipoint stress with weak symmetry).

**Why here.** These are the *other* discretization of the same physics Phase 27
solves with VEM. Porting them after `ad-mechanics` means the physics is already
validated and only the discretization is new - the two can be cross-checked
against each other on the same problem, which is a strong test neither has alone.

**Depends on.** Phase 25 core grid/utility leftovers, Phase 27.

**Validation.** MATLAB dumps; plus the cross-discretization check above - VEM and
MPSA must agree on Terzaghi/Mandel to discretization error, and disagree in a way
that shrinks under refinement.

**Size / risk.** `L`. `mpsaw` is 91 files and much of it is assembly plumbing;
scope it to the discretization plus one validated case, per the project's
established "vertical slice" discipline.

## Phase 29 - `dual-continuum-mech` + `fv-unsat` `M`

**What.** Dual-porosity poromechanics (38 files: constitutive coefficients,
shape factors, transfer) and unsaturated flow / Richards' equation (18 files).

**Why here.** Both are small, both sit directly on top of what Phases 27-28 and
the existing `dual_porosity` module provide. `fv-unsat` also unlocks `re-mpfa`
(Richards with MPFA, 7 files) almost for free.

**Depends on.** Phases 27, 28; existing `dual_porosity`.

**Validation.** MATLAB dumps. `fv-unsat` additionally has analytic Richards
solutions and a mass-conservation invariant.

**Size / risk.** `M`. Low.

**Progress.** Phases 396-397 establish the `fv-unsat` prerequisite in MRST's
`data/`, `models/`, and `utility/` layout. The hydraulic-property catalog and
both van Genuchten-Mualem parameterizations are ported with analytic moisture
capacities. `modelRE` and `solverRE` provide a fully implicit backward-Euler
Richards step over the Phase-30 MPFA operators, including arithmetic face
relative permeability, gravity-head support, sources, and MRST-compatible
`Q`/`psiEq` entry points. The nontrivial infiltration test closes every cell
to `9.0e-16` and the global balance to `1.4e-15`. Upstream weighting and the
coupled unsaturated-Biot model remain; the flow foundation needed by
  `re-mpfa` is now runnable.

---

# Tier 3 - Discretizations

TPFA is currently the only scheme in PyMRST. That is a real limitation: TPFA is
only consistent on K-orthogonal grids, and the unstructured grids Phase 23/26
now make available are exactly where it stops being valid. This tier fixes the
inconsistency the meshing work exposes.

## Phase 30 - `mpfa` (+ `nfvm`, `re-mpfa`) `M`

**What.** `computeMultiPointTrans`, `incompMPFA`, and the O-method local
assembly (18 files). Then `nfvm` (nonlinear FV, 24 files) and `re-mpfa` (7).

**Progress.** The first MPFA vertical slice is implemented as
`pymrst/modules/mpfa` (Phase 387): the legacy local-flux-mimetic MPFA-O
transmissibility operators and a single-phase Dirichlet pressure solve, with
snake-case and MRST-compatible public names. `validate_phase387.py` applies the
discriminating linear patch test on a smoothly twisted grid with a full
permeability tensor: MPFA is exact to roundoff while TPFA has a visible error.
Phase 388 extends the solve with harmonic total-mobility weighting, mixed
pressure/flux boundary conditions (unspecified faces are no-flow), cell
sources, conservative face/subface flux reconstruction, pure-Neumann
null-space handling, and single-phase gravity. Its tests pin local mass
conservation, prescribed boundary rate, mobility scaling, and the hydrostatic
patch. Wells, tensor assembly for general 3-D corners, `nfvm`, and `re-mpfa`
remain.

Phase 389 adds rate/BHP wells and restores the MRST module shape around the
kernel: public `computeMultiPointTrans` / `incompMPFA`, the
`private/incompMPFAlegacy` state/fluid path, and the corresponding `utils`
block/local-flux helpers. The rate-injector/BHP-producer test closes to
`2.6e-14`, pins BHP exactly, and verifies public/private entry-point parity.
The remaining MPFA item is tensor assembly for general 3-D corners; `nfvm` and
`re-mpfa` remain separate follow-on modules.

Phase 390 completes that MPFA item with the stabilized general-corner tensor
assembly and the MRST `private/computeMultiPointTransTensorAssembly` /
`private/incompMPFATensorAssembly` dispatch. A square bipyramid provides the
discriminating topology: its four-face apex is rejected by the legacy path,
while tensor assembly reproduces a full-permeability-tensor linear patch to
`1.8e-15` and conserves each cell to `1.3e-27`. On ordinary nice corners the
legacy and tensor paths agree to `3.6e-15`.

**The primary MPFA capability slice is complete.** MRST's blockwise large-model
assembly is still an optimization gap (the numerical result is the same), and
its alternate `nicecorner` / `directinverse` shortcuts are not separate code
paths. The separately named `nfvm` and `re-mpfa` follow-on modules remain.

Phase 391 starts the `nfvm` follow-on in the matching
`nfvm/solvers/utils` layout: `findHAP`, two-dimensional `findOSflux`,
`TransNTPFA`, `computeFlux`, and the `NTPFA` gradient object. On Cartesian
grids the harmonic averaging points equal face centroids exactly in 2-D and
3-D. On a smoothly twisted 2-D grid with a full permeability tensor, the
complete nonlinear one-sided flux chain reproduces the linear Darcy flux on
unaffected interior stencils to `3.6e-14`. Remaining NFVM work: automatic HAP
correction, 3-D one-sided stencil decomposition, `AvgMPFA`, `Hybrid`, and the
AD-model setup functions.

Phase 392 adds `correctHAP` and the three-dimensional `findABC` decomposition.
The explicit correction obeys MRST's face-radius constraint exactly without
changing interpolation weights; on a distorted 3-D grid with a full tensor,
NTPFA reproduces the interior linear Darcy flux to `2.5e-14`.

Phase 393 adds `AvgMPFA` and `AvgTransNTPFA` in their MRST locations. The fixed
arithmetic blend preserves the twisted-grid full-tensor linear patch to
`3.1e-14`, with exact class/utility parity. `Hybrid` and AD-model setup remain.

Phase 394 adds `Hybrid`, `setup_gdz`, and the MRST model installers
`setNTPFADiscretization`, `setAvgMPFADiscretization`, and
`setHybridDiscretization`. Face-block replacement and nonlinear gravity
differences match their direct constructions exactly. AvgMPFA propagates AD
Jacobians through its sparse maps; NTPFA propagates them in MRST's lagged mode
(nonlinear weights frozen at the current pressure). Fully implicit
pressure-dependent NTPFA weights remain the one functional NFVM gap.

Phase 395 closes that gap. `TransNTPFA` now assembles both correction terms as
sparse pressure maps, so the convex quotient is an AD expression and carries
the full derivative of the pressure-dependent weights. Threshold decisions use
current numeric values (the same semismooth branch convention as MRST). On a
distorted anisotropic grid, the AD Jacobian matches central finite differences
with `4.7e-09` maximum and `9.2e-10` relative error. Implicit and lagged
residual values agree to `2.8e-14`, while their Jacobians differ materially,
proving the nonlinear derivative is active.

Phases 396-397 then supply the `fv-unsat` dependency for the final `re-mpfa`
follow-on: van Genuchten-Mualem laws plus a conservative, fully implicit
Richards/MPFA model and time-step solver.

Phase 398 ports the complete numerical contents of `re-mpfa/functions`
directly—`arithmeticMPFA_hydHead`, `upstreamMPFA_hydHead`, `newtonSolver`, and
`timeStepping`—under the matching module layout. This is an implementation,
not a delegating wrapper. A hydraulic-head operator adapter exposes the
already-ported MPFA matrices as the `F`, `boundF`, and `div` operators used by
the original algorithms. Both upstream directions are pinned exactly, all
adaptive-step branches are exercised, and the ported Newton-AD infiltration
step converges in six iterations to a `5.1e-14` residual.

**The functional NFVM discretization slice is complete.** Plot helpers, cached
stencil serialization, and MEX hull acceleration are non-functional
optimizations/workflow conveniences.

**Why here.** MPFA is the standard answer to TPFA's grid-orthogonality
requirement, it is small, and it is the discretization the fracture and
geomechanics modules assume when the grid is not orthogonal.

**Depends on.** Phase 25 (`multiEig`), Phase 23 (something non-orthogonal to run
it on).

**Validation.** MATLAB dumps. Plus the discriminating physical test: on a
deliberately non-K-orthogonal grid, MPFA reproduces a linear pressure field
exactly and TPFA does not. That test is worth more than the dump, because it is
the entire reason the module exists.

## Phase 31 - `mimetic`, `vag`, `vem`, `dg` `L`

**What.** Mimetic FD (14), vertex approximate gradient (6), VEM flow (19),
discontinuous Galerkin transport (77).

**Why here.** Completeness of the discretization shelf. `dg` is the big one and
the most valuable (high-order transport kills the smearing that first-order
upwind imposes on every saturation front PyMRST currently computes).

**Depends on.** Phase 25 (`faceFlux2cellVelocity`), Phase 30.

**Validation.** MATLAB dumps; plus Buckley-Leverett, where dG's whole claim is a
sharper front than Phase 3's upwind solver against the same analytic solution.

**Size / risk.** `L`, mostly `dg`. Consider splitting `dg` into its own phase.

**Progress.** Phase 399 ports `computeMimeticIP` at the MRST module root,
including the `ip_simple`, `ip_tpf`, `ip_quasitpf`, `ip_rt`, `ip_quasirt`, and
`ip_qfamily` families; hybrid, mixed, TPFA, and composite assembly modes; and
optional face transmissibilities. On a distorted grid with a full permeability
tensor, the defining affine-flux consistency identity holds to `8.0e-15`.

Phase 400 ports the first functional `incompMimetic` solve plus MRST's
deprecated `solveIncompFlow` entry point in their original locations. The
hybrid system eliminates half-face fluxes, solves cell/interface pressures,
handles pressure/flux boundaries and sources, preserves every cell equation
for pure-Neumann systems, and reconstructs conservative global flux. Mixed-BC
cell conservation is `1.2e-19`; balanced-source conservation is `3.6e-21`.
Phase 401 completes the gravity and well branches of this vertical slice.
Gravity enters through the original half-contact centroid vectors and
mobility-weighted mimetic operator; rate and BHP wells are assembled as
additional hybrid contacts with Peaceman indices and completion pressure
drops. A full-tensor distorted-grid hydrostatic patch has `1.2e-21` maximum
flux, while the rate-injector/BHP-producer case balances to `2.1e-18`.
The alternate hybrid-system solver modes under `modules/mimetic/utils/` are now
ported (Phase 402) - `schur_complement_symm`, `mixed_symm`, `tpf_symm` and the
fault variant `schur_complement_symm_fault`, the three equivalent ways
`incompMimetic` reduces and solves the mimetic saddle-point `B v = f + C p - D
lam`, `C'v = g`, `D'v = h`. The Schur (hybrid) mode is pinned to MRST's
`schurComplementSymm` on identical inputs from a real mimetic discretisation
(`validate_phase402.py`): cell pressure to 1.8e-15, half-face flux to 2.6e-26,
face pressures to 5.3e-15. `mixed_symm`/`tpf_symm` are validated by exact
saddle-point reconstruction, and `tpf_symm == mixed_symm` up to MRST's
opposite multiplier-sign convention (a faithful difference between the two MRST
functions, not a bug). `schur_complement_symm_fault` reduces to the Schur mode
when its fault-coupling `E` is zero. **Phase 31 mimetic core is complete.**

**VEM (Phase 403) - foundation started.** VEM builds its local projections from
moments (integrals of monomials over each cell), so it needs exact quadrature on
general polygons. The utility layer under `modules/vem/utils/` is ported:
`poly_dim` (dimension of the degree-`k` polynomial space), `triangle_quad_rule`
(Gauss rules of precision 1/2/3/7), and `polygon_int`, which integrates a
function over polygonal cells by triangulating each and mapping the reference
rule onto every triangle (cell-node connectivity derived from the cell's faces,
since PyMRST has no direct field). `validate_phase403.py` matches MRST's
`polygonInt` on the monomial basis `[1,x,y,x^2,xy,y^2]` to 8.9e-16, and the rules
integrate degree-`k` polynomials exactly (x^6 to 1/7).

The **3-D quadrature (Phase 404)** completes the integration layer:
`tetrahedron_quad_rule` (precision 2/3 on MRST's equilateral reference tet) and
`polyhedron_int`, which Delaunay-tetrahedralises each cell and affinely maps the
reference rule onto every tet. `validate_phase404.py` matches MRST's
`polyhedronInt` on the 3-D monomial basis to 8.9e-16.

A **working first-order VEM pressure solve** (`vem_poisson`) is now in place and
passes the defining **patch test**: on a smoothly-twisted (non-orthogonal) grid
it reproduces a linear pressure field to `7.1e-15` at every node - which TPFA
cannot. It is the standard lowest-order nodal (H1-conforming) VEM: consistency
part exact for linear pressures via the projection onto linear polynomials, plus
a stability part that vanishes on them; each local matrix is SPD and annihilates
constants.

The **mixed VEM inner product (Phase 405)** now matches MRST's `computeVirtualIP`
number for number. `compute_virtual_ip(g, rock, k=1)` produces the `S` structure -
the per-cell local stiffness `A_E = PiNstar' Mc PiNstar + sigma_E (I-PiN)'(I-PiN)`
and the node DOF map - following MRST's face-based `computeBD2D`: the constant row
of `B` is **edge-length weighted** (not the vertex average), the gradient rows
carry the full permeability tensor, and the `ip_simple` stability is
`sigma_E = h_E^(dim-2) tr(K)/dim`. `incomp_vem` assembles the blocks and solves.
`validate_phase405.py` compares the per-cell local-stiffness *eigenvalues*
(an ordering-independent fingerprint of the inner product) against MRST on a
twisted grid with a full permeability tensor: **max eigenvalue error 1.8e-27**
(scale 2.76e-12, i.e. ~15 digits). The patch test holds for scalar and full-tensor
permeability. First-order (k=1, nodal DOFs) is ported; the k=2 scheme (edge and
cell moments), the 3-D face projections and the MPFA/TPFA flux-reconstruction
options of `computeVirtualIP`, plus the rest of `incompVEM` (Neumann BC, wells,
flux reconstruction), remain.

**Remaining Phase 31 discretizations (scope note).** `vag`, the rest of `vem`,
and `dg` are each large numerical-method ports. **VAG** (606 lines) is built
entirely on MRST's `IndexArray`/`TensorMap` indexing-table DSL (`tbl_utils`),
which PyMRST does not have - a faithful port needs that framework first, or a
from-scratch reimplementation of the vertex-gradient scheme. **VEM** is ~2000
lines of polynomial-projection assembly (foundation now in place). **DG** is 77
files of high-order transport - the most valuable (sharper Buckley-Leverett
fronts than first-order upwind) and the largest; the roadmap already suggests
splitting it into its own phase.

---

# Tier 4 - Physics breadth

These are the modules that turn a reservoir simulator into *the* reservoir
simulator. Each is a set of state functions once Phase 24 exists - which is
precisely the argument for Phase 24.

## Phase 32 - `ad-eor` + `solvent` `L`

**What.** Polymer, surfactant, and their combination (75 files); four-phase
solvent/miscible gas (16 files).

**Why here.** The most-used EOR physics, and the cleanest demonstration that the
state-function graph pays off: in MRST these are almost entirely new state
functions plugged into an unchanged black-oil skeleton.

**Depends on.** Phase 24. Hard dependency - without the graph these are large
bespoke ports.

**Validation.** MATLAB dumps on the modules' own examples; plus conservation
invariants (polymer mass, surfactant adsorption) and the physical signature that
distinguishes them (viscosity multiplier, IFT reduction, residual-oil
mobilization).

## Phase 33 - `ad-scal` + `geochemistry` + `steady-state` `L`

**What.** Special core analysis / relative-permeability estimation (99 files),
geochemical reactions and equilibrium (37), steady-state upscaling (61).

**Why here.** All three are "one more physics on top of the graph" and none
blocks anything else. `steady-state` extends the existing `upscaling` module.

**Depends on.** Phase 24; `steady-state` also on existing `upscaling`.

**Validation.** MATLAB dumps. `geochemistry` additionally has charge- and
mass-balance invariants that catch a wrong equilibrium solve immediately.

## Phase 34 - Compositional flow (depth on `compositional`) `M`

**What.** PyMRST now has the `OverallCompositionCompositionalModel` core:
Peng-Robinson flash, differentiated flash, flow properties, fully implicit
overall-composition flow, component-aware no-water wells, oil/gas gravity
fluxes, `model.water = true` closed-domain flow, and water injector/BHP
producer wells, capillary pressure phase-potential shifts, and the
`opt.pressure` reduced pressure-equation assembly plus pressure-only Newton
solve, MATLAB-validated no-well sequential transport for both no-water and
water-active cases, and MATLAB-validated sequential well transport for both
no-water and water-active paths with MRST-style pressure/transport well-stream
handoff. The sequential pressure/transport wrapper now has a native
multi-control schedule regression. Remaining depth is full MRST parity:
MATLAB-backed validation of richer sequential schedules, richer facility-tree
controls, and eventually deeper natural-variable variants.

**Why here.** The expensive half (the EOS and flash) is already ported and
validated to 1e-12. This is the cheap half, and it converts a standalone
calculator into a simulator.

**Depends on.** Phase 24; existing `compositional`.

**Validation.** MATLAB dumps through Phase 60: closed 1D equilibration,
component-aware injector/producer wells, closed vertical gravity segregation,
closed water/oil/gas flow, water injector/BHP producer wells, and capillary
pressure flow, plus direct `PressureOverallCompositionModel` pressure-equation
assembly and pressure-only Newton step; plus component-mass conservation
invariants.

---

# Tier 5 - Fractures

## Phase 35 - `hfm_v2` (pEDFM) + `dfm` `L`

**What.** Projection-based EDFM (180 files, the successor to the `hfm` already
ported) and discrete fracture matter (63 files, conforming fracture meshes).

**Why here.** `hfm` (2D EDFM) is ported and matches MATLAB to machine precision.
pEDFM fixes EDFM's known failure on low-permeability (sealing) fractures - a real
correctness improvement, not just more surface area. `dfm` needs conforming
meshes, which is why it waits for Phase 26.

**Depends on.** Phase 25 (`processNNC`, `makeInternalBoundary`), Phase 26 (`dfm`),
existing `hfm`.

**Validation.** MATLAB dumps. Plus the discriminating case: a sealing fracture,
where EDFM is known to leak and pEDFM is not.

## Phase 36 - `shale` + `hwu-fractures` + `dual-porosity-permeability` `M`

**What.** Shale-specific physics (48 files: sorption, Knudsen diffusion,
non-Darcy), Heriot-Watt fracture models (31), DPDP (8).

**Depends on.** Phase 35; existing `dual_porosity`.

**Validation.** MATLAB dumps; plus the sorption isotherm and diffusion limits.

---

# Tier 6 - Solvers and scale

## Phase 37 - `linearsolvers` + `msrsb` + `msfvm` + `msmfem` `XL`

**What.** AMG/CPR preconditioning and iterative solvers (19 files); multiscale
restriction-smoothed basis (69), multiscale finite volume (16), multiscale mixed
FE (41).

**Why here.** Everything in PyMRST currently ends in `scipy.sparse.linalg.spsolve`
- a direct solve. That is fine for validation cases and fatal for field scale.
This tier is what makes the port *fast* rather than merely *correct*, and it is
placed after the physics because a fast wrong answer is worthless and the
physics phases are what establish correctness.

**Depends on.** Phase 24 (the solver protocol).

**Validation.** Correctness is easy and not the point: the multiscale solution
must converge to the fine-scale solution already validated. The real test is
**scaling** - solve time versus problem size, against the direct solver, on SPE10
(Phase 39). A multiscale solver that is correct but not faster has failed.

**Size / risk.** `XL`. High risk: this is the phase where Python's performance
ceiling actually bites, and it may need `scipy`'s AMG (`pyamg`) or a compiled
kernel rather than a faithful port. Expect to depart from MRST's implementation
here more than anywhere else, and say so.

> **CPR is done** (shipped as Phase 43; see the README). `ad_core.CPRSolver` -
> AMG on the decoupled pressure system, ILU on the remainder, GMRES around the
> pair - plus `select_linear_solver`, MRST's `selectLinearSolverAD`. The
> iteration count is flat (13-24) across a 200x range in problem size, so the
> field-scale ceiling this phase existed to remove is removed: 432,000 cells
> solve where the direct solve could not go, and 98,000 cells go 3.3x faster.
>
> Two predictions in the paragraphs above were wrong and are worth keeping
> visible. **The performance ceiling did not bite**: `pyamg` carried the elliptic
> stage and no compiled kernel was needed. And the *hard* part was not the
> algorithm - CPR is textbook, and the decoupling is one line. It was that
> `scipy`'s `spilu` defaults are actively wrong for this job: SuperLU's COLAMD
> column permutation destroys the grid locality an incomplete factorisation
> depends on, and the resulting factor is singular. That cost more time than
> everything else in the phase combined, and no amount of reading MRST would
> have predicted it, because MATLAB's ILU does not make that choice.
>
> **Still open in this phase:** the multiscale solvers (`msrsb`, `msfvm`,
> `msmfem`) and `domain-decomposition`. Those remain `XL`, and the scaling
> argument for them is now weaker than it was when this was written, because
> CPR already scales.

## Phase 38 - `domain-decomposition` + `agglom` `M`

**What.** Additive Schwarz / overlapping domain decomposition (20 files);
grid agglomeration and coarsening heuristics (41).

**Depends on.** Phase 37; existing `upscaling`.

**Validation.** MATLAB dumps; convergence to the undecomposed solution.

---

# Tier 7 - Data, workflow, and the long tail

## Phase 39 - Datasets and cases: `spe10`, `sleipner`, `test-suite` `M`

**What.** SPE10 (15 files - the standard heterogeneity benchmark), Sleipner (9 -
the real CO2 storage dataset with 20 years of monitoring), and `test-suite` (the
standard case library MRST's own tests run on).

**Why here.** These are *validation infrastructure*, and the argument for doing
them earlier is real. They are here because they are useful rather than blocking,
and because SPE10's main value (a hard problem for a linear solver) lands with
Phase 37. Sleipner is the natural capstone for the co2lab work, which is the most
complete part of the port.

**Validation.** The datasets are the reference. SPE10's published solutions;
Sleipner's observed plume outlines.

## Phase 40 - `h2lab` + `hybrid-ve` `M`

**What.** Hydrogen storage (15 files - only `h2props` is currently sampled);
hybrid VE/3D models (15 - VE where the plume is thin, 3D where it is not).

**Depends on.** Existing `co2lab`.

**Validation.** MATLAB dumps. `hybrid-ve` has a strong internal check: it must
reproduce the pure-VE answer where the VE assumption holds and the pure-3D answer
where it does not.

## Phase 41 - Wells and workflow: `wellpaths`, `nwm`, `network-models`, `postprocessing`, `static-modeling`, `heterogeneity` `L`

**What.** Deviated/multilateral well paths (12), near-well modelling (51),
GPSNet-style network models (13), post-processing (10), static modelling (12),
heterogeneity generation (8).

**Why here.** Workflow rather than physics; nothing depends on them. `nwm` is the
substantial one and depends on Phase 26 (near-well grids are PEBI grids).

**Validation.** MATLAB dumps throughout.

## Phase 42 - Meshing bindings: `gmsh`, `triangle` `S`

**What.** Two and three files respectively - thin readers/writers for external
meshers.

**Why here.** Trivial once Phase 23's constructors exist: the job is to parse the
mesher's output into `triangle_grid` / `tetrahedral_grid`. Python has better
bindings available (`meshio`, `triangle`) than MRST's shell-out, and this is one
of the few places where departing from the MATLAB implementation is clearly
right.

**Validation.** Round-trip a mesh through both codes and compare the grids.

---

# Tier 8 - Beyond the port: an optional GPU / autodiff backend

Everything above is a port: the reference is MATLAB, and "correct" means "matches
MRST". This tier is the one deliberate exception - a capability MRST does not
have, carried over as an *idea* (not code) from an earlier prototype in this same
workspace, `PyRevSimAD`, which put MRST grid data onto `torch` tensors with the
intent of GPU-accelerated, machine-learning-facing simulation. It is last because
it depends on the whole stack being correct first, and it is optional because the
`scipy` path remains the validation ground truth.

## Phase 43 - An optional `torch` array backend `XL`

**What.** A backend seam under `utils/adi.py`, not a rewrite. Today `ADI` carries
a value plus a list of `scipy.sparse` Jacobian blocks - forward-mode AD, CPU,
exact. The change is to let the array/linear-algebra layer be pluggable:

- a `scipy` backend (the default, unchanged - it is what every `validate_phaseN`
  checks against), and
- a `torch` backend, where state vectors are `torch.Tensor` on a chosen device
  and differentiation is `torch.autograd` (reverse-mode) rather than hand-carried
  sparse blocks.
- the model code (`ad_blackoil`, `geothermal`, the Phase 24 state-function graph)
  is written once against the seam and runs on either.

**Why here.** Two payoffs the port itself cannot give:

1. **Speed / scale.** Phase 37 flags that everything ends in
   `scipy.sparse.linalg.spsolve` and that Python's performance ceiling bites at
   field scale. A `torch` backend puts the linear algebra on the GPU without a
   bespoke C kernel, and is the least-disruptive way to get there.
2. **Differentiable simulation.** Reverse-mode autograd through the whole
   simulator makes gradient-based history matching and well optimisation
   end-to-end differentiable (complementing, not replacing, the analytic adjoint
   of Phase 18 and the ES-MDA of the ensemble module), and lets the simulator
   train or be replaced by ML/GNN surrogates - the original intent of
   `PyRevSimAD`, whose reference case was literally named `..._gnn_...`.

**Depends on.** Phase 24 (the solver/model protocol is what gets a second
backend), Phase 18 (the adjoint is the gradient oracle to check against). Should
follow Phase 37, whose scaling harness is the benchmark this phase has to move.

**Validation.** No MATLAB reference exists - this is a Tier-8 "physics/invariant"
check per the contract, on three levels:

1. **Differential against the validated backend.** The `torch` backend must
   reproduce the `scipy` backend's already-MATLAB-validated states on the Phase
   4/5/5b cases, to a stated tolerance (float64; expect ~1e-12, looser than the
   direct MATLAB checks because the linear solves differ).
2. **Gradient agreement.** Reverse-mode `torch.autograd` gradients must match the
   Phase-18 analytic adjoint and finite differences - the same 8e-10-relative bar
   the adjoint already meets.
3. **Scaling, not just correctness.** Like Phase 37, the point is speed: solve
   time vs. problem size on GPU against the `scipy` direct solve on SPE10. A
   backend that is correct but not faster at scale has not earned its place.

**Size / risk.** `XL`, and the risks are real and worth naming. (a) *Faithfulness
drift* - a second backend is where a "port" quietly becomes "a framework of its
own", so the seam must be narrow and the `scipy` path must stay bit-for-bit what
it is today. (b) *Precision* - float64 throughout or the validation tolerances
are gone; `PyRevSimAD` cast to float32, which is exactly what PyMRST cannot do.
(c) *Sparse-on-GPU* is still the weak spot of the torch ecosystem; the honest
fallback is dense-block or CPU-torch where sparse GPU solves underperform, and to
say so. (d) It may be better as a *sibling package* that imports PyMRST than as a
hard dependency, to keep `numpy`/`scipy` the only requirement for the core.

---

# Declared out of scope

Not everything in the tree should be ported, and saying so explicitly is part of
the plan:

| Module | Why not |
|---|---|
| `book` (125 files) | Textbook example scripts, not library code. Port individual examples if they make good validation cases. |
| `jutul` | A bridge to a Julia simulator. Meaningless in a Python port; the equivalent would be a fresh binding, not a port. |
| `libgeometry` | MEX-accelerated geometry. `compute_geometry` is already vectorised NumPy; the C is a MATLAB workaround. |
| `mrst_api`, `mrst-gui` | MATLAB session plumbing and MATLAB GUI. `mrst-gui`'s useful part (`plotToolbar`) is already ported into `pymrst/plotting`. |
| `co2lab-demo`, `co2lab-legacy` | Demo scripts and a superseded implementation of code already ported. |
| `MICPMechModel` | Not portable: `equationsMICPMech.m` is absent from the MRST tree (it lived in the deleted `ad-micp-mech` module). Already recorded in `GAPS.md`. |
| MATLAB session utils | `mrstModule`, `mrstPath`, `merge_options`, ... - no Python equivalent needed. |

---

# Ordering, in one picture

```
  Phase 24  ad-core state-function graph  ──┬─→ 27 ad-mechanics ─→ 28 fvbiot/mpsaw ─→ 29 dc-mech/fv-unsat
     (DONE - unblocked Tier 3-6)            │
                                            ├─→ 32 ad-eor/solvent
                                            ├─→ 33 ad-scal/geochem/steady-state
                                            ├─→ 34 compositional flow
                                            └─→ 37 linearsolvers/multiscale ─→ 38 dd/agglom
  Phase 25  core leftovers  ────────────────┬─→ 30 mpfa ─→ 31 mimetic/vag/vem/dg
     (M, unblocks 30, 35)                   └─→ 35 hfm_v2/dfm ─→ 36 shale/hwu/dpdp
  Phase 26  upr meshing  ───────────────────┬─→ 35 dfm
     (L, completes Phase 23)                └─→ 41 nwm
  Independent: 39 datasets · 40 h2lab/hybrid-ve · 41 workflow · 42 mesh bindings
  Beyond the port: 43 torch/GPU backend (after 37; gradient-checked against 18)
```

Three phases (24, 25, 26) gate almost everything else. They are, respectively,
the biggest single piece of work in the project, a sweep of small mechanical
functions, and the other half of the phase just completed.

# The recommended next phase

**Phases 24 and 27 are done.** The graph landed, and geomechanics landed on
top of it - and the second confirmed the first: poroelasticity reduced to a
single state function, exactly as the fork above predicted.

Of what remains, the two gating items are **Phase 25** (core leftovers -
`processNNC`, `multiEig`, `sliceGrid` and friends; cheap, mechanical, easy
MATLAB dumps, and it unblocks the MPFA and fracture phases) and **Phase 26**
(`upr` meshing - the other half of Phase 23, since `pebi` consumes point sets
and nothing here produces a good one).

For visible new capability, the strongest candidates are now:

- **Phase 32** (`ad-eor`: polymer and surfactant) - the cleanest test of the
  Phase-24 thesis on physics that is *purely* new state functions, and the
  most-used EOR physics there is.
- **Phase 28** (`fvbiot` / `mpsaw`) - the other discretization of the physics
  Phase 27 just validated, so the two can be cross-checked against each other
  on Terzaghi and Mandel. Cheaper than it looks, now that the physics is
  pinned by an analytic benchmark rather than only by MATLAB - which, in this
  corner of the tree, has now been shown to be wrong.

Three phases in, the pattern the graph predicted is holding: Phase 27's
single-phase poroelasticity was an `L`, and its multiphase extension was
~150 lines. Physics that MRST expresses as state functions is now cheap to
port; physics that does not exist in the graph (new discretizations,
solvers) is not. That is the right shape for the remaining roadmap.
