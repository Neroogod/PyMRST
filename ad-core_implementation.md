# ad-core: implementation roadmap to full MRST parity

`README.md` records what is built. `GAPS.md` records what is missing across the
whole port. This file is narrower and deeper: it is the plan for finishing
**one module - `ad-core`** - to the point where MRST's `ad-core/` and PyMRST's
`ad_core/` are the same library, function for function, in the same directory
layout.

It is written after the layout refactor that emptied the `ad_core` package
root: every implementation now lives in the MRST-shaped subdirectory, and the
root holds only `__init__.py`. That refactor closed the *shape* gap. This file
plans the *depth* gap.

## The layout contract

MRST's `ad-core/` has **no files at its root**; every symbol lives in a
subdirectory. PyMRST mirrors this directory-for-directory, with two mechanical
transforms that are forced by Python and applied everywhere:

- hyphens become underscores (`ad-core` -> `ad_core`);
- MATLAB's one-function-per-file convention collapses into grouped modules
  where that reads better, but **never across a subdirectory boundary**. A
  symbol MRST files under `statefunctions/flux/` lands under
  `ad_core/statefunctions/flux/`, not anywhere else.

Every phase below names its MRST source files and its exact PyMRST destination
path. A phase is not done until the code sits at that path.

## The validation contract (inherited, non-negotiable)

Every phase is proved the way the existing 336 were (see `ROADMAP.md`):

1. **A MATLAB reference.** `dump_phaseN_reference.m` runs in the MRST tree next
   door and saves `.mat` arrays; `validate_phaseN.py` rebuilds the same case
   and compares field by field. Tolerances have run 1e-13 to 1e-16.
2. **Fix the inputs, compare the algorithm** where an upstream difference
   (triangulator, RNG, solver tolerance) would otherwise pollute the check.
3. **Physics where MATLAB cannot be the judge** - an invariant that breaks if
   the code is wrong (a finite-difference gradient, a conservation law, an
   analytic solution). State which, and why.
4. **Self-tests in the file**, run by `python ad_core/.../module.py` and
   collected by `tests/test_all.py`.
5. **Report what the reference gets wrong.** Seven MRST defects have surfaced
   this way; "matches MATLAB" and "correct" are not the same claim.

Sizes are `S` (a session), `M` (a few), `L` (substantial), `XL` (a project).

---

## Where ad-core stands

Counting library symbols only (excluding `examples/`, `book-ii/`, the MEX
operator layer, `sim_runner/` packed-problem infra, and the MATLAB `plotting/`
GUI - all classified "out of scope" below):

| Subdirectory | present / MRST | what is missing |
|---|---|---|
| `backends` | 5 / 7 | two diagonal Jacobian variants + operator helpers |
| `initialization` | 2 / 14 | equilibration regions, deck/ECLIPSE init |
| `models` | 2 / 4 | `GenericReservoirModel`, `WrapperModel` |
| `models/aquifers` | 2 / 2 | - |
| `models/composite` | 0 / 1 | `CompositeModel` |
| `models/facilities` | 6 / 15 | MS wells, perforation packing, control eqns |
| `models/facilities/separators` | 3 / 3 | - |
| `models/facilities/vfp` | 1 / 1 | - |
| `models/wells` | 0 / 6 | `WellModel`, control switching, connection DP |
| `simulators` | 1 / 4 | the three adjoint drivers (folded in `optimization`) |
| `solvers` | 1 / 14 | the solver class hierarchy + base classes |
| `statefunctions` | 5 / 7 | `FacilityFlowDiscretization`, `StateFunctionDependent` |
| `statefunctions/components` | 2 / 3 | `GenericComponent` |
| `statefunctions/composite` | 0 / 2 | `CouplingTerm`, `SourceCouplingTerm` |
| `statefunctions/facility` | 0 / 11 | every well/perforation statefunction |
| `statefunctions/flowprops` | 8 / 23 | the whole `BlackOil*` family + reduction factors |
| `statefunctions/flowprops/hysteresis` | 0 / 2 | `HystereticRelativePermeability` |
| `statefunctions/flowprops/virtual` | 0 / 1 | `SaturationProperty` |
| `statefunctions/flux` | 9 / 14 | TPFA-as-statefunction, thresholded potential |
| `statefunctions/flux/flow_state` | 0 / 4 | explicit / implicit / AIM builders |
| `statefunctions/flux/upwind` | 0 / 5 | the upwind + WENO hierarchy |
| `statefunctions/utils` | 0 / 4 | graph introspection + plotting |
| `timesteps` | 5 / 5 | - |
| `upscale` | 0 / 3 | model/schedule/state upscaling |
| `utils` | ~1 / 80 | see the honest note below |

**The `utils` row overstates the gap.** MRST files 80 loose functions there;
of those, roughly a third are MATLAB session plumbing with no Python analogue,
another third are already implemented as *methods* on PyMRST classes rather
than standalone functions (convergence checks live on the models, `faceUpstr`
lives in the operators), and the genuine-capability remainder is what Tier 7
below ports. File-count parity is the wrong metric here; capability parity is
the target.

**The one structural fact that orders everything:** MRST's physics modules
(`ad-blackoil`, `ad-eor`, `solvent`, `compositional`) are *written as
statefunctions plugged into a grouping*. PyMRST has the grouping machinery (the
spine) and the generic evaluators, but almost none of the **specializations** -
the `BlackOil*` family, the upwind hierarchy, the well statefunctions. Today
black-oil PVT works, but through `ad_blackoil`'s own backend code, not through
graph-resident statefunctions. Until the specialization layer exists, each new
physics module is a bespoke residual; once it exists, each is a close port.
That is why Tier 1 is first, and why it is most of the value.

---

## Tier 1 - the statefunction specialization layer

This tier is the reason the graph was built. It is largely independent of the
other tiers and should land first.

### AC1 - Flux discretization statefunctions ~~[DONE - Phase 337]~~
- **What:** `TwoPointFluxApproximation`, `PermeabilityPotentialGradient`,
  `PermeabilityGradientDiscretization`, `PhasePotentialDifferenceThresholded`,
  `FaceComponentMobility`.
- **Path:** `ad_core/statefunctions/flux/`.
- **Why here:** the flux group is the root of the flow half of the graph;
  every model's conservation equation reads through it. `TwoPointFluxApproximation`
  in particular is the statefunction wrapper over the operators PyMRST already
  has - a thin, high-certainty first port that pins the pattern.
- **Depends on:** nothing new (the operators and `PhaseFlux` exist).
- **Validation:** reproduce an existing oil-water flux assembly (Phase 5/24)
  through the new statefunctions; bit-for-bit against the current path.
- **Size/risk:** M. Low risk - the physics is already validated, this re-homes
  it in the graph.
- **Status:** landed in `ad_core/statefunctions/flux/`
  (`permeability_gradient_discretization.py`, `permeability_potential_gradient.py`,
  `phase_potential_difference_thresholded.py`). `validate_phase337.py` proves
  the new discretization-agnostic path reproduces the validated `PhaseFlux`
  exactly (`-FaceMobility * (K grad Phi)`, val 1.7e-18 on a 1.5e-2 flux scale)
  and that the thresholded potential reduces to the parent at zero `THPRES` and
  blocks sub-threshold flow. **`FaceComponentMobility` is deferred** to after
  AC2 (`UpwindProperty`/`UpwindFunctionWrapperDiscretization`) and AC6
  (`ComponentMobility`), which it depends on - it is filed here in MRST but
  cannot stand until those land.

### AC2 - The upwind hierarchy ~~[DONE - Phase 338 core + Phase 348 WENO]~~
- **What:** `UpwindDiscretization` (base), `UpwindProperty`,
  `UpwindFunctionWrapperDiscretization`, `WENOUpwindDiscretization`,
  `getBoundaryFacesMerged`.
- **Path:** `ad_core/statefunctions/flux/upwind/`.
- **Why here:** `PhaseFlux` currently hardcodes single-point upwind; the
  hierarchy makes upwinding a pluggable statefunction, which WENO and the
  higher-order schemes need. `getBoundaryFacesMerged` is a shared helper.
- **Depends on:** AC1.
- **Validation:** single-point upwind reproduces the current result exactly;
  WENO against MRST's `wenoExampleAD` dump on a 1D Buckley-Leverett front.
- **Size/risk:** M. WENO reconstruction is the fiddly part.
- **Status:** the core landed in `ad_core/statefunctions/flux/upwind/`
  (`upwind_discretization.py` - base + `UpwindFunctionWrapperDiscretization`;
  `upwind_property.py` - the mixin; `boundary_faces.py` -
  `get_boundary_faces_merged`). `validate_phase338.py` proves the wrapper
  reproduces the validated `FaceMobility` upwinding exactly (val 0.0, jac 0.0)
  through the `UpwindProperty` mixin. With the mixin in place, AC1's
  `FaceComponentMobility` is now unblocked on the upwind side (it still waits on
  AC6's `ComponentMobility`). The mirror-cell geometry `get_boundary_faces_merged`
  is ported and invariant-tested. **`WENOUpwindDiscretization` now DONE
  (Phase 348):** the 1D WENO reconstruction (`flux/upwind/weno.py`) - two
  smoothness-weighted linear stencils per face, ported from MRST's
  `getFaceValue1D` / `interpolate1D`. Validated against a real MATLAB dump of
  MRST's own `WENOUpwindDiscretization` (`dump_phase348_weno.m` ->
  `ref_p348_weno.mat`, run in MRST 2024a via MATLAB R2021b): PyMRST reproduces
  MRST's face reconstruction to 1.1e-16 on a smooth ramp and to 7.9e-31 on a
  shock front, in both upwind directions. (A real bug surfaced doing this:
  MRST's boundary marker is the index 0, a valid cell in PyMRST's 0-based
  scheme; the port uses a -1 sentinel.) The 2D/3D triangulation-based WENO
  (`getFaceValueTri`) is a separate, larger port.

### AC3 - Flow-state builders ~~[DONE - Phase 339]~~
- **What:** `FlowStateBuilder` (base), `ExplicitFlowStateBuilder`,
  `ImplicitFlowStateBuilder`, `AdaptiveImplicitFlowStateBuilder`.
- **Path:** `ad_core/statefunctions/flux/flow_state/`.
- **Why here:** these decide which quantities are evaluated at the old vs new
  time level - the switch that makes a model IMPES, fully-implicit, or AIM.
  The sequential module and explicit transport need them.
- **Depends on:** AC1, AC2.
- **Validation:** implicit builder reproduces the fully-implicit result;
  explicit builder against an IMPES transport step; AIM against MRST on a case
  with a mixed implicit/explicit cell partition.
- **Size/risk:** M.
- **Status:** landed in `ad_core/statefunctions/flux/flow_state/`
  (`flow_state_builder.py` - base + implicit + the `seed_property`/
  `owning_grouping` mechanism; `explicit_flow_state_builder.py`;
  `adaptive_implicit_flow_state_builder.py`). All three hybridize by
  pre-seeding the graph's property cache, which maps exactly onto MRST's
  container seeding. `validate_phase339.py`: the implicit builder reproduces
  the fully-implicit `PhaseFlux` to 0.0; the explicit builder reproduces the
  IMPES split ``-T*FaceMobility(state0)*PhasePotentialDifference(state)`` to
  0.0 (val and jac); AIM blends cell mobilities per-cell to 0.0. **The
  CFL-driven step limit / cell mask is AC21** (`estimateSaturationCFL` /
  `estimateCompositionCFL`) and is injected or supplied directly for now; the
  hybridization mechanism itself is complete.

### AC4 - The BlackOil PVT flowprops ~~[PVT core DONE - Phase 340; relperm/pc deferred]~~
- **What:** `BlackOilDensity`, `BlackOilViscosity`, `BlackOilShrinkageFactors`,
  `BlackOilPoreVolume`, `BlackOilCapillaryPressure`, `RsMax`, `RvMax`,
  `SurfaceDensity`, `BaseRelativePermeability`, `DensityDerivedShrinkageFactors`.
- **Path:** `ad_core/statefunctions/flowprops/`.
- **Why here:** this is the headline gap. It moves black-oil PVT out of
  `ad_blackoil`'s private backend and into graph statefunctions, so
  `GenericBlackOilModel` composes exactly as MRST's does - and so `ad-eor` and
  `solvent`, which subclass these, have a base to stand on.
- **Depends on:** AC1 (density feeds the potential gradient).
- **Validation:** SPE1 and SPE9 PVT terms against the existing black-oil dumps,
  field by field; the assembled residual must match the current `ad_blackoil`
  residual to machine precision before the old backend path is retired.
- **Size/risk:** L. The `Rs`/`Rv` saturation switches are where MRST's own
  edge cases live.
- **Status:** the **PVT core** landed in `ad_core/statefunctions/flowprops/`
  on MRST's base/`BlackOil*` hierarchy: `shrinkage_factors.py`, `viscosity.py`,
  `pore_volume.py`, `surface_density.py`, `density.py` (`BlackOilDensity` +
  `DensityDerivedShrinkageFactors`), `saturation_limits.py` (`RsMax`/`RvMax`),
  with `blackoil_pvt_utils.py` centralising the saturated-state read.
  `validate_phase340.py` proves each against the pre-split, MATLAB-validated
  PVT on the live-oil SPE 1 model at a mixed saturated/under-saturated state:
  shrinkage/viscosity/pore-volume to 0.0, density to 1.1e-13, all Jacobians to
  <=1.7e-16; and `DensityDerivedShrinkageFactors == Density/rhoS` to 1e-16
  (recovering `b` exactly for the carry-free water phase). The black-oil
  validators phase24/phase47 still pass. **Deferred within AC4:**
  `BaseRelativePermeability` and `BlackOilCapillaryPressure` - the relative-
  permeability hierarchy (WOG/OG/WO/WG variants + endpoint scaling) is a
  distinct sub-effort belonging to the `FlowPropertyFunctions` group, best done
  as its own phase. **`BaseRelativePermeability` now DONE (Phase 345)** - the
  WO / WOG phase-by-phase hierarchy with the connate-water three-phase oil
  blend, in `flowprops/base_relative_permeability.py` on the
  `flowprops/virtual/SaturationProperty` base, reproducing the validated
  `RelativePermeability` to 0.0 on both oil-water and three-phase black-oil
  (SPE 1). Residual follow-on: `BlackOilCapillaryPressure` and ECLIPSE endpoint
  scaling (`rock.krscale`, which PyMRST's rock does not carry).

### AC5 - Pressure-reduction factors ~~[analytic DONE - Phase 341; numerical deferred]~~
- **What:** `PressureReductionFactors`, `BlackOilPressureReductionFactors`,
  `NumericalPressureReductionFactors`.
- **Path:** `ad_core/statefunctions/flowprops/`.
- **Why here:** these are the weights that collapse the component system to a
  pressure equation - what CPR and the sequential module use. PyMRST's CPR
  currently derives them internally; this exposes them as MRST does.
- **Depends on:** AC4.
- **Validation:** the weights against MRST on SPE1; the CPR iteration count
  must be unchanged.
- **Size/risk:** M.
- **Status:** the two analytic classes landed in
  `flowprops/pressure_reduction_factors.py` (`PressureReductionFactors` -
  ``pv/rho``; `BlackOilPressureReductionFactors` - the live-oil/wet-gas weights
  with the ``1/b_o - rs/b_g`` coupling). `validate_phase341.py` proves the
  defining true-IMPES invariant, which needs no external reference:
  ``sum_i w_i M_i`` has (to round-off) no derivative w.r.t. the secondary
  unknowns - immiscible ``d/dsW`` ~1e-16 relative, and on live-oil SPE 1
  ``sum_i w_i M_i = 1.0000`` exactly with both the ``sW`` and ``rs`` Jacobian
  blocks vanishing to <=4e-16 (the dissolved-gas cross terms cancel exactly
  between the oil and gas components). **Deferred:**
  `NumericalPressureReductionFactors` (weights differentiated numerically from
  the mass Jacobian - tied to the diagonal-AD backend and the sequential
  solver's `reductionFactorProps`) and the ``useUndersaturated`` diagonal-AD
  refinement of the black-oil weights.

### AC6 - Component flowprops and the generic component ~~[DONE - Phase 342]~~
- **What:** `ComponentMobility`, `ComponentPhaseDensity`, `ComponentProperty`,
  `GenericComponent`.
- **Path:** `ad_core/statefunctions/flowprops/` and
  `ad_core/statefunctions/components/`.
- **Why here:** completes the component abstraction so a component can carry its
  own mobility and density - required by compositional-native and by any tracer
  or reactive species.
- **Depends on:** AC4.
- **Validation:** against the compositional component dumps already in the tree.
- **Size/risk:** M.
- **Status:** landed. The MRST `GenericComponent` method surface
  (`get_component_density` / `get_component_mobility` / `get_component_mass`) is
  now on PyMRST's `Component` base (`statefunctions/components/components.py`),
  with `GenericComponent` added; `flowprops/component_properties.py` adds the
  `ComponentProperty` mixin, `ComponentPhaseDensity` and `ComponentMobility`
  (= phase mobility x component density). **This also closes AC1's deferred
  `FaceComponentMobility`** (`flux/face_component_mobility.py`), which needed
  both AC2's upwind mixin and AC6's `ComponentMobility`. `validate_phase342.py`
  proves the loop: `ComponentMobility == Mobility * ComponentPhaseDensity` to
  0.0, and `scale_c * FaceComponentMobility * (K grad Phi)` reproduces the
  validated `ComponentPhaseFlux` to 3.5e-18 on a 2.3e-2 flux scale (single-point
  upwind gives ``upw(a)*upw(b) = upw(a*b)``, so the component-mobility route and
  the density-upwind route are the same number).

### AC7 - Facility statefunctions (wells in the graph) ~~[core DONE - Phase 343; component-well layer deferred]~~
- **What:** `FacilityWellMapping`, `WellIndex`, `WellPhaseFlux`,
  `PerforationMobility`, `PerforationPressureGradient`,
  `PerforationComponentPhaseDensity`, `WellComponentPhaseFlux`,
  `WellComponentTotalFlux`, `WellComponentTotalFluxDensityMix`,
  `WellComponentTotalVolumeBalanceCrossflow`, `InjectionSurfaceDensity`, and the
  `FacilityFlowDiscretization` grouping.
- **Path:** `ad_core/statefunctions/facility/` (grouping in
  `ad_core/statefunctions/`).
- **Why here:** the well equations are currently assembled procedurally inside
  the 2,868-line `facility_model.py`. MRST expresses them as statefunctions in
  a facility grouping. This is the single largest coherent gap in the module,
  and it is what makes crossflow, component wells, and multisegment wells
  compose instead of branch. `WellIndex` exists today only as the params
  function `compute_well_index`; this is its statefunction form.
- **Depends on:** AC4, AC6.
- **Validation:** the multi-perforation well column (Phase 49) and the
  compositional wells (Phases 54-57) reproduced through the facility grouping,
  field by field, before the procedural path is retired.
- **Size/risk:** L. Crossflow (`WellComponentTotalVolumeBalanceCrossflow`) is
  the subtle one.
- **Status:** the **well-flux core** landed in
  `ad_core/statefunctions/facility/` as a real state-function grouping
  (`FacilityFlowDiscretization`) hosted on a light `FacilityContext` - the graph
  counterpart of MRST's facility model, adapted to PyMRST's well structs:
  `facility_well_mapping.py`, `well_index.py`,
  `perforation_pressure_gradient.py`, `perforation_mobility.py`,
  `well_phase_flux.py` (with `calculate_phase_rate`). `validate_phase343.py`
  proves `WellPhaseFlux` reproduces the procedural `_single_well` reservoir flux
  ``cq_vol`` on two five-perforation wells (one injector, one producer) to
  4.3e-19 on a 3.4e-3 rate scale - both the producer branch (reservoir mobility)
  and the injector branch (total mobility x injection composition). Phase 49
  (the multi-perforation wellbore column) still passes. **Deferred - the
  component-well / crossflow-mixture layer:** `WellComponentPhaseFlux`,
  `WellComponentTotalFlux`, `WellComponentTotalFluxDensityMix`,
  `WellComponentTotalVolumeBalanceCrossflow`, `PerforationComponentPhaseDensity`
  and `InjectionSurfaceDensity`, plus the full `crossFlowMixture` (crossflow
  into a producer / production from an injector). These build on the phase flux
  landed here and are the intricate part; they are a focused follow-on.

### AC8 - Hysteresis and virtual saturation ~~[DONE - Phase 345]~~
- **What:** `HystereticRelativePermeability`, `evaluateFunctionCellSubsetReg`,
  `SaturationProperty`.
- **Path:** `ad_core/statefunctions/flowprops/hysteresis/` and
  `.../flowprops/virtual/`.
- **Why here:** PyMRST has `KilloughGasHysteresis` as a standalone; this folds
  it into the MRST relperm-statefunction hierarchy and adds the region-subset
  evaluator that CO2 and WAG cases need. `SaturationProperty` is the virtual
  base several relperm variants inherit.
- **Depends on:** AC4 (`BaseRelativePermeability`).
- **Validation:** a drainage-imbibition loop against MRST; the existing Killough
  self-test must still pass.
- **Size/risk:** M.
- **Status:** landed with the relperm hierarchy that unblocked it.
  `SaturationProperty` in `flowprops/virtual/saturation_property.py` (connate
  water; endpoint scaling flagged inactive, as PyMRST's rock has no
  `krscale`); `HystereticRelativePermeability` in
  `flowprops/hysteresis/hysteretic_relative_permeability.py`, extending
  `BaseRelativePermeability` and overriding its gas hook (`evaluate_gas_rel_perm`)
  with PyMRST's `KilloughGasHysteresis` against the state's `sGMax`.
  `validate_phase345.py` proves the two invariants: on drainage (``sG == sGMax``)
  the hysteretic gas relperm equals the base curve; on imbibition
  (``sG < sGMax``) it is trapped strictly below the drainage curve (max
  reduction 4.6e-2 on SPE 1). **Residual follow-on:**
  `evaluateFunctionCellSubsetReg` (the multi-region saturation-function
  evaluator) is not yet needed - PyMRST evaluates a single saturation region -
  and lands with multi-region relperm; the AD Jacobian is dropped on the
  imbibition cells, matching MRST's numeric scanning-curve evaluation.

### AC9 - Composite coupling terms ~~[core DONE - Phase 347, with AC12]~~
- **What:** `CouplingTerm`, `SourceCouplingTerm`, `StateFunctionDependent`.
- **Path:** `ad_core/statefunctions/composite/` (and
  `StateFunctionDependent` in `ad_core/statefunctions/`).
- **Why here:** the mechanism by which two physics groupings (flow +
  geomechanics, flow + thermal) share a term. PyMRST's `ad_mechanics` couples
  by hand today; this is the general machinery, and it is what `CompositeModel`
  (AC12) is built on.
- **Depends on:** AC1-AC7 (needs real statefunctions to couple).
- **Validation:** reproduce the fixed-stress poroelastic coupling (Phase 27)
  through a `CouplingTerm`; the surface-uplift invariants must hold.
- **Size/risk:** M.
- **Status:** landed with AC12 (Phase 347), which provided the composite host.
  `statefunctions/composite/coupling_term.py`: `CouplingTerm` (the base
  source-insertion mechanism - `insert_coupling` adds ``sign * q`` into named
  submodel equations), `SourceCouplingTerm` (a conservative transfer between
  two submodels sharing component names: sign -1 out of A, +1 into B), and the
  `StateFunctionDependent` marker. `validate_phase347.py` proves the defining
  property - conservation: a `SourceCouplingTerm` A->B removes exactly ``q``
  from A's residual and adds exactly ``q`` to B's, leaving ``eq_A + eq_B``
  unchanged to 0.0. The full flow-geomechanics coupling *solve* against
  `ad_mechanics` remains the follow-on (it needs geomechanics as a
  state-function grouping - see AC12).

### AC10 - Graph introspection and plotting ~~[DONE - Phase 344]~~
- **What:** `buildStateFunctionDigraph`, `getStateFunctionGroupingDependencyGraph`,
  `plotStateFunctionGroupings`, `printStateFunctionGroupingTikz`.
- **Path:** `ad_core/statefunctions/utils/`.
- **Why here:** not physics - tooling. It walks the dependency graph to draw or
  print it, which is how MRST documents and debugs a grouping. Cheap once the
  groupings above exist.
- **Depends on:** AC1-AC9 (there must be a graph to walk).
- **Validation:** the extracted dependency edges against a hand-checked
  grouping; the plot is a rendering test, not a numeric one.
- **Size/risk:** S.
- **Status:** landed in `statefunctions/utils/dependency_graph.py`:
  `get_state_function_grouping_dependency_graph` (the whole-model cross-grouping
  graph, nodes ``"struct.name"``, edges ``dep -> dependent``, primary variables
  as ``state.<var>`` inputs), `build_state_function_digraph` (adds a global
  topological order + cycle check), `ancestors_of` (a node's evaluation
  subtree), and `to_dot` / `format_grouping_text` (Graphviz DOT + text, no
  plotting backend needed). `validate_phase344.py` proves the structural
  invariants: completeness (every grouping node present), acyclicity with an
  order respecting every cross-grouping edge, and - the tie to real behaviour -
  the nodes the solver actually evaluates for a target are a subset of that
  target's graph ancestors (14 evaluated within 16 ancestors for
  `ComponentTotalFlux`).

---

## Tier 2 - models

### AC11 - Generic and wrapper reservoir models ~~[DONE - Phase 346]~~
- **What:** `GenericReservoirModel`, `WrapperModel`.
- **Path:** `ad_core/models/`.
- **Why here:** `GenericReservoirModel` is the model that assembles purely from
  a statefunction grouping (no hand-written equations) - the payoff of Tier 1.
  `WrapperModel` adapts one model to another's interface, used by sequential and
  multiscale.
- **Depends on:** Tier 1.
- **Validation:** `GenericReservoirModel` reproduces `GenericBlackOilModel` on
  SPE1 with no bespoke residual code.
- **Size/risk:** M.
- **Status:** landed. `models/generic_reservoir_model.py` - the marker base
  (MRST's `GenericReservoirModel` is an empty mix-in) with `get_model_equations`
  as the generic graph-assembly entry point; `GenericBlackOilModel` is made an
  instance of it (`class GenericBlackOilModel(GenericReservoirModel)`), the same
  is-a relationship MRST declares. `models/wrapper_model.py` - `WrapperModel`,
  a transparent delegator to a `parent_model` (explicit delegation of the
  driver surface + a `__getattr__` catch-all), the base the sequential /
  multiscale families subclass. `validate_phase346.py` proves the black-oil
  model is-a `GenericReservoirModel` and its `get_model_equations` equals its
  `conservation_equations` to 0.0 (value + Jacobian), and that a `WrapperModel`
  reproduces the parent's properties and equations to 0.0 while a one-method
  subclass keeps the rest delegated. The inheritance change is regression-clean
  (85 targeted self-tests pass; black-oil validators 24/47/49 unchanged).

### AC12 - CompositeModel ~~[core DONE - Phase 347; coupled solve deferred]~~
- **What:** `CompositeModel`.
- **Path:** `ad_core/models/composite/`.
- **Why here:** the model that runs two coupled groupings, the general form of
  what `ad_mechanics` does by hand. Depends on the coupling terms.
- **Depends on:** AC9, AC11.
- **Validation:** the coupled poromechanics case rebuilt as a `CompositeModel`;
  match Phase 27/33.
- **Size/risk:** L.
- **Status: core landed.** `models/composite/composite_model.py` - a
  `CompositeModel` built from named submodels, each keeping its own sub-state;
  `get_model_equations` asks each submodel for its equations on its sub-state
  and applies every registered `CouplingTerm` (AC9). `validate_phase347.py`
  proves the composition is faithful (a one-submodel composite yields the
  submodel's own equations to 0.0; a two-submodel composite yields both
  concatenated) and the coupling is exactly conservative. **Deferred - the
  coupled solve:** MRST's full `CompositeModel` (784 lines) also packs all
  submodel primary variables into one Newton system, splits the linearized
  problem, and runs the coupled solve; and its headline validation (flow +
  geomechanics matching Phase 27/33) needs **geomechanics expressed as a
  state-function grouping**, which PyMRST's hand-coupled `ad_mechanics` is not.
  Re-expressing mechanics as a grouping is a large piece beyond ad-core; until
  it exists, the coupled *solve* against `ad_mechanics` cannot be validated to
  this project's bar, so the composition + conservation core is what lands here.

---

## Tier 3 - wells and facility completion

### AC13 - The well model and control switching
- **What:** `WellModel`, `setupWellControlEquations`,
  `setupWellControlEquationsSingleWell`, `updateSwitchedControls`,
  `updateSwitchedWellControls`, `updateConnectionDP`.
- **Path:** `ad_core/models/wells/`.
- **Why here:** PyMRST has control switching inside `facility_model.py`
  (Phase 39) but not as MRST's separable `WellModel` with standalone control
  equations. This factors it out to the MRST layout so a well type is a class,
  not a branch. `setupWellControlEquationsSingleWell` already has a MATLAB
  regression (Phase 120) to extend.
- **Depends on:** AC7.
- **Validation:** the Phase 39/120 control-switch regressions, re-driven through
  `WellModel`.
- **Size/risk:** M.

### AC14 - Multisegment wells and perforation packing
- **What:** `MultisegmentWell` (class), `setupMSWellEquationSingleWell`,
  `combineMSwithRegularWells`, `convert2MSWell`, `nozzleValve`,
  `packPerforationProperties`, `unpackPerforationProperties`,
  `reorderWellPerforationsByDepth`, `computeWellContributionsSingleWell`,
  `computeWellContributionsNew`.
- **Path:** `ad_core/models/facilities/`, `ad_core/models/wells/`.
- **Why here:** PyMRST has `assemble_multisegment_equations` (a function) and a
  params-level `multisegment.py`, but not the MRST `MultisegmentWell` class with
  the nozzle/valve pressure-loss models and the perforation pack/unpack
  round-trip. Completes multisegment to MRST's surface.
- **Depends on:** AC13.
- **Validation:** the Phase 109 MS-well packed-variable check, extended to a
  segmented producer against an MRST dump.
- **Size/risk:** L.

---

## Tier 4 - solvers

### AC15 - The linear-solver class hierarchy
- **What:** `LinearSolverAD` (base), `LinearizedProblem`, `BackslashSolverAD`,
  `NoOpSolverAD`, `HandleLinearSolverAD`, `GMRES_ILUSolverAD`,
  `getNonLinearSolver`, `selectLinearSolverAD`.
- **Path:** `ad_core/solvers/`.
- **Why here:** PyMRST has `CPRSolver` and `DirectSolver` as concrete classes
  but not MRST's base `LinearSolverAD` / `LinearizedProblem` contract the models
  program against, nor the plain Backslash/NoOp/GMRES-ILU siblings and the
  `getNonLinearSolver` factory. This makes the solver layer swappable the way
  MRST's is.
- **Depends on:** nothing new.
- **Validation:** each concrete solver reproduces the current CPR/direct result;
  `LinearizedProblem` assembly matches MRST's block layout.
- **Size/risk:** M. **Out of scope:** `AGMGSolverAD`, `AMGCLSolverAD` and the
  `AMGCL_CPR*` variants - external C bindings, constant-factor wins, tracked as
  not-portable in `GAPS.md`.

---

## Tier 5 - initialization

### AC16 - Equilibration regions
- **What:** `getInitializationRegionsBase`, `getInitializationRegionsBlackOil`,
  `getInitializationRegionsCompositional`, `getInitializationRegionsDeck`,
  `getMinMaxPhaseSaturations`, `getMinMaxPhaseSaturationsFromRelPerm`,
  `getMinimumPhaseSaturations`.
- **Path:** `ad_core/initialization/`.
- **Why here:** PyMRST has `initializeEquilibriumPressures/Saturations` and
  `init_state_equil` (Phase 42) but not the *region* layer that lets a field
  have several equilibration regions with different contacts - which every real
  deck has.
- **Depends on:** AC4 (needs the PVT statefunctions for region densities).
- **Validation:** a two-region equilibration against an MRST dump; contact
  depths and the transition-zone saturation profile field by field.
- **Size/risk:** M.

### AC17 - Deck and ECLIPSE initialization
- **What:** `initStateBlackOilAD`, `initStateDeck`, `initEclipseProblemAD`,
  `getDeckOPMData`, `plotRegionContacts`.
- **Path:** `ad_core/initialization/`.
- **Why here:** the end-to-end path from a parsed deck to an initial AD state,
  which is how MRST starts SPE1/SPE9 from their `.DATA` files rather than
  hand-built states.
- **Depends on:** AC16, and `deckformat` (already present).
- **Validation:** SPE1 initial state from the deck vs the hand-built state used
  today; identical to machine precision.
- **Size/risk:** M.

---

## Tier 6 - adjoint simulators (MRST-layout homing)

### AC18 - Home the adjoint drivers under ad_core/simulators
- **What:** `computeGradientAdjointAD`, `computeGradientPerturbationAD`,
  `computeSensitivitiesAdjointAD`.
- **Path:** `ad_core/simulators/`.
- **Why here:** these exist in PyMRST - folded into the `optimization` module
  (`optimization/utils/compute_gradient_adjoint_ad.py`, `optimization/adjoint.py`).
  MRST files them under `ad-core/simulators/`. This phase gives them their MRST
  home (as re-exports or a move) and broadens the adjoint from the current
  flow-rate objective to the general parameter-sensitivity form MRST's
  `computeSensitivitiesAdjointAD` supports.
- **Depends on:** Tier 1 (a graph-assembled model makes the adjoint general).
- **Validation:** the existing Phase 18 gradient check (adjoint vs
  finite-difference) re-run through the `ad_core/simulators` entry point;
  add a perturbation-vs-adjoint cross-check.
- **Size/risk:** M. Mostly relocation + generalization, not new math.

---

## Tier 7 - utils (the genuine capability, not the plumbing)

### AC19 - Convergence
- **What:** `CNV_MBConvergence`, `checkWellConvergence`,
  `getConvergenceValuesCNV`, `getConvergenceValuesWells`,
  `printConvergenceReport`.
- **Path:** `ad_core/utils/`.
- **Why here:** PyMRST checks CNV/material-balance convergence *inside* the
  model classes. MRST factors it into standalone functions the `NonLinearSolver`
  calls, so any model gets the same report. Factoring it out matches the layout
  and removes duplicated logic across the black-oil models.
- **Depends on:** nothing new.
- **Validation:** the CNV/MB values on an SPE1 step against MRST's printed
  report, number for number.
- **Size/risk:** M.

### AC20 - Schedule tooling
- **What:** `makeScheduleConsistent`, `convertDeckScheduleToMRST`,
  `compressSchedule`, `refineSchedule`, `filterSchedule`, `combineSchedules`,
  `simpleSchedule`, `convertReportToSchedule`.
- **Path:** `ad_core/utils/`.
- **Why here:** the tools that build and reshape a schedule. `simpleSchedule`
  and `convertDeckScheduleToMRST` are the common entry points; the rest are
  editing operations real multi-year field schedules need.
- **Depends on:** `deckformat`.
- **Validation:** SPE9's schedule from its deck vs the hand-built schedule;
  `compressSchedule`/`refineSchedule` round-trip invariants.
- **Size/risk:** M.

### AC21 - CFL estimates and discretization setters
- **What:** `estimateSaturationCFL`, `estimateCompositionCFL`,
  `setTimeDiscretization`, `setWENODiscretization`, `setMPFADiscretization`.
- **Path:** `ad_core/utils/`.
- **Why here:** the CFL estimates drive explicit/AIM timestep and cell
  selection (they pair with AC3); the setters are the one-call switches that
  reconfigure a model's flux discretization. `setMPFADiscretization` is a stub
  until MPFA lands (tracked in `GAPS.md`), but the hook belongs here.
- **Depends on:** AC3.
- **Validation:** the saturation CFL against MRST on a Buckley-Leverett case;
  the setters against the assembled discretization they produce.
- **Size/risk:** M.

### AC22 - Numeric utility functions
- **What:** `getMultiDimInterpolator`, `interpolateIDW`, `fastInterpTable`,
  `getMultipliers`, `faceUpstr`, `phaseDensitiesTobfactor`, `splitFaceCellValue`,
  `expandMatrixToCell`, `expandIfUniform`, `mergeOrderedArrays`,
  `calculatePhaseRate`, `convertReservoirFluxesToSurface`, and the remaining
  genuinely-numeric helpers.
- **Path:** `ad_core/utils/`.
- **Why here:** shared numeric primitives several of the phases above reach for.
  Port them as they are needed by a consuming phase rather than in a block, and
  give each a self-test.
- **Depends on:** consumed by AC1-AC21 as needed.
- **Validation:** per-function self-tests; MATLAB cross-check where a consuming
  phase already dumps the surrounding case.
- **Size/risk:** M, spread across the tiers.

### AC23 - I/O, result handling, standalone solve, model selection
- **What:** `ResultHandler`, `HandleStruct`, `getReportOutput`,
  `getWellOutput`, `standaloneSolveAD`, `selectModelFromDeck`,
  `getReservoirModel`/`setReservoirModel`.
- **Path:** `ad_core/utils/`.
- **Why here:** `selectModelFromDeck` is the factory that turns a parsed deck
  into the right model class - the top of the deck-to-simulation path.
  `standaloneSolveAD` is the one-call solve. `ResultHandler` is MRST's on-disk
  state store.
- **Depends on:** AC17, the model tiers.
- **Validation:** `selectModelFromDeck` picks the black-oil model for SPE1 and
  the compositional model for a compositional deck; `standaloneSolveAD`
  reproduces a scripted solve.
- **Size/risk:** M.

---

## Tier 8 - backend performance internals

### AC24 - Diagonal Jacobian variants and operator helpers
- **What:** `ConservationLawJacobian`, `FixedWidthJacobian`,
  `getSparseBlocksFixedWidth`, `blockCSRtoSparse`,
  `discreteDivergenceDiagonalJac`, and the `backends/diagonal/utils` helpers
  (`diagMult`, `diagProductMult`, `double2GenericAD`, `getSparseArguments`,
  `getSparseBlocks`, `incrementSubset`, `initVariablesAD_diagonal*`,
  `matrixDims`).
- **Path:** `ad_core/backends/diagonal/`, `.../diagonal/operators/`,
  `.../diagonal/utils/`.
- **Why here:** performance, not capability. PyMRST's diagonal backend already
  works with a row-major Jacobian and Numba kernels; these are the remaining
  MRST Jacobian layouts (fixed-width, conservation-law) and the block-assembly
  helpers that make large-model assembly cheaper. Port when assembly cost, not
  correctness, is the ceiling.
- **Depends on:** nothing functionally; benchmarked against the current backend.
- **Validation:** identical assembled Jacobian to the current backend on a
  mid-size model; a wall-clock benchmark to justify each variant.
- **Size/risk:** L. **Out of scope:** the MEX operator layer
  (`backends/diagonal/operators/mex/`, 17 files + 12 `.cpp`) - MATLAB C
  acceleration with no place in a NumPy/Numba backend.

---

## Tier 9 - upscaling

### AC25 - Model, schedule, and state upscaling
- **What:** `upscaleModelTPFA`, `upscaleSchedule`, `upscaleState`.
- **Path:** `ad_core/upscale/`.
- **Why here:** coarsening an AD model + its schedule + its state onto a coarse
  grid, the AD-side companion to the flow-based upscaling PyMRST already has in
  the `upscaling` module. Independent of the rest.
- **Depends on:** `upscaling` (present), Tier 1.
- **Validation:** an upscaled SPE10 slab against MRST's coarse model response.
- **Size/risk:** M.

---

## Explicitly out of scope (not a port target)

These are counted out of the parity numbers above, on the same grounds
`GAPS.md` uses:

- **The MEX operator layer** - `backends/diagonal/operators/mex/` (17 `.m`
  wrappers + 12 `.cpp`). MATLAB C acceleration; the PyMRST diagonal backend is
  already vectorized NumPy with Numba kernels.
- **External linear-solver bindings** - `AGMGSolverAD`, `AMGCLSolverAD`,
  `AMGCLSolverBlockAD`, `AMGCL_CPRSolverAD`, `AMGCL_CPRSolverBlockAD`. C library
  bindings; constant-factor wins, not new capability. CPR already scales.
- **Packed-problem infrastructure** - `simulators/sim_runner/` (12 files):
  `PackedProblemManager`, background simulation, on-disk problem packing. Batch
  orchestration, not simulation physics.
- **The MATLAB plotting GUI** - `plotting/` (`plotWellSols`, `simpleUIbar`,
  `simulationRuntimePanel`, ...). PyMRST's interactive surface is the PyVista
  `plot_toolbar`; these are MATLAB `uicontrol` panels.
- **Examples and the book-ii tutorials** - `examples/` (12) and
  `examples/book-ii/` (13). Not library code. Port *individually* when one makes
  a good validation case (as several already have).
- **MATLAB session plumbing in `utils/`** - `obj2hash`, `numelData`,
  `getGridSYMRCMOrdering`, `readSummaryLocal`, and the like: helpers for
  MATLAB's runtime with no Python analogue.

---

## Ordering in one picture

```
  Tier 1  statefunction specialization layer  ─────────────┐  (the keystone)
    AC1 flux ─ AC2 upwind ─ AC3 flow-state                 │
    AC4 BlackOil PVT ─ AC5 reduction ─ AC6 components       │
    AC7 facility SFs ─ AC8 hysteresis ─ AC9 coupling        │
    AC10 graph utils                                        │
                                                            ├─→ Tier 2 models
                                                            │     AC11 generic/wrapper
                                                            │     AC12 composite
                                                            ├─→ Tier 3 wells (AC13-14)
                                                            ├─→ Tier 6 adjoint (AC18)
                                                            └─→ Tier 9 upscale (AC25)
  Tier 4 solvers (AC15) ── independent
  Tier 5 initialization (AC16-17) ── needs AC4
  Tier 7 utils (AC19-23) ── mostly independent, feed the tiers above
  Tier 8 backend perf (AC24) ── independent, performance only
```

**Tier 1 gates the module.** It is also where most of the value is: it is the
layer `ad-eor`, `solvent`, and compositional-native are waiting on, and the
layer that turns "PyMRST computes this" into "MRST code reads across
unchanged." The recommended first phase is **AC1** - the smallest, most certain
piece, and the one that pins the pattern for the fourteen statefunction phases
behind it.

## Sequencing against the global phase counter

The project numbers phases globally (the tree is at Phase 336). These `AC`
labels are local to this document for dependency clarity; each becomes the next
global `Phase N` when it is taken up, with its `dump_phaseN_reference.m` /
`validate_phaseN.py` pair, exactly as the contract requires.

## A note on the pre-existing test failures

Independent of this roadmap, five validation cases currently fail in files this
plan does not start from: `phase19` (co2lab atlas - imports a missing
`co2lab.atlas._atlas_dir`), `phase89` (vemmech calibration), and
`phase312`/`313`/`317` (facility model). They predate the ad-core layout
refactor and should be triaged before Tier 3 (which builds on the facility
model) begins.
