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

### AC12 - CompositeModel ~~[DONE - Phase 347 core + Phase 349 coupled solve]~~
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
  concatenated) and the coupling is exactly conservative. **The coupled solve
  now DONE (Phase 349):** `assemble_coupled_system` builds the global Newton
  system - each submodel's residual and Jacobian (via `combine_equations`) as a
  diagonal block, plus the coupling off-diagonal blocks - and
  `LinearPressureCoupling` is a concrete conservative inter-model transfer that
  couples two reservoir submodels through pressure. `validate_phase349.py`
  proves it with two oil-water submodels: the system is square, the cross-block
  A->B is exactly the analytic ``+coeff`` (zero when uncoupled), the coupling
  Jacobian is finite-difference-correct (``dR_A/dp_B`` matches to 1.2e-21), the
  coupling conserves total mass, and one coupled Newton step reduces ``||R||``
  (5.7e-3 -> 1.8e-3). This is the general coupled-solve machinery, validated
  without geomechanics. **Residual follow-on:** the *specific* flow-geomechanics
  validation (matching Phase 27/33) still needs geomechanics re-expressed as a
  state-function grouping - a large piece beyond ad-core - so that headline case
  is the one deferred, not the coupled-solve capability itself.

---

## Tier 3 - wells and facility completion

### AC13 - The well model and control switching ~~[core DONE - Phase 350]~~
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
- **Status:** the control layer landed in `ad_core/models/wells/`:
  `well_control_equations.py` (`setup_well_control_equation_single_well` - the
  bhp / rate / wrat / orat / grat / lrat control equation with MRST's
  shut-well and singular-phase fallbacks) and `update_switched_controls.py`
  (`update_switched_controls` / `violated_limit` - switch a well to the limit
  it breaks). `validate_phase350.py` proves both against the MATLAB-validated
  procedural facility path: the control equation reproduces
  `FacilityModel._control_equation` to 0.0 (value + Jacobian) for every control
  type, and `update_switched_controls` switches exactly the wells
  `FacilityModel.update_switched_controls` does, to the same modes (a rate
  injector -> bhp, a bhp producer -> orat). **Residual follow-on:**
  `updateConnectionDP` (PyMRST has it as `connection_pressure_drop`) and a full
  `WellModel` *class* wrapper are a thin further factoring; `computeWellContributionsNew`
  is the legacy well-contribution path superseded by the facility statefunctions
  (AC7).

### AC14 - Multisegment wells and perforation packing ~~[core DONE - Phase 351]~~
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
- **Status:** the multisegment-well machinery was already on the MRST layout and
  MATLAB-validated - `MultisegmentWell` / `WellSegment`
  (`params/wells_and_bc/multisegment.py`), `assemble_multisegment_equations` /
  `MultisegmentEquationResult`
  (`ad_core/models/facilities/multisegment/`), checked against MRST's converted
  MS-well layout in Phase 109 (which still passes). This phase adds the missing
  utility that MS wells rely on: `reorder_well_perforations_by_depth`
  (`ad_core/models/facilities/reorder_well_perforations_by_depth.py`), a
  faithful port of MRST's `reorderWellPerforationsByDepth`.
  `validate_phase351.py` proves its invariants (a deterministic depth sort needs
  no external reference): perforations end depth-ordered, every per-perforation
  field (`cells`/`WI`/`dZ`/`cstatus`) is permuted by the same order, and the
  reorder is idempotent. **Residual follow-on:** the MRST-internal
  `packPerforationProperties` / `unpackPerforationProperties` (explicitly
  "internal use in FacilityModel"), the nozzle/valve pressure-loss models, and
  `convert2MSWell` / `combineMSwithRegularWells` - PyMRST's MS wells already
  carry the segment friction law (Phase 109), so these are surface-completion
  items, not new capability.

---

## Tier 4 - solvers

### AC15 - The linear-solver class hierarchy ~~[core DONE - Phase 352]~~
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
- **Status:** landed. `solvers/linear_solver_ad.py` - the `LinearSolverAD` base
  (the ``solve_linear_system(A,b)`` contract, a `solve(jac,rhs,model)` entry
  compatible with PyMRST's existing solvers, optional Jacobi scaling, a solve
  report) and `LinearizedProblem` (matrix + rhs + named primary-variable blocks,
  with `store_increments` splitting a solved increment back into them).
  `solvers/concrete_solvers.py` - `BackslashSolverAD` (direct `spsolve`),
  `NoOpSolverAD` (zero increment), `GMRES_ILUSolverAD` (restarted GMRES + ILU).
  `validate_phase352.py` proves them on a real 126-unknown black-oil Newton
  Jacobian: `BackslashSolverAD` solves ``A dx = rhs`` to 1.3e-17 and returns the
  *same* increment as the validated `DirectSolver` (0.0), `GMRES_ILUSolverAD`
  converges to it (5e-7), `NoOpSolverAD` returns zero, and `LinearizedProblem`
  splits the increment - and the CPR anchor (Phase 43) still passes.
  **Residual follow-on:** `HandleLinearSolverAD` (a function-handle wrapper) and
  the `getNonLinearSolver` / `selectLinearSolverAD` factories (PyMRST has
  `select_linear_solver`) are thin surface items.

---

## Tier 5 - initialization

### AC16 - Equilibration regions ~~[core DONE - Phase 353]~~
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
- **Status:** the region layer landed in `initialization/regions.py`:
  `get_initialization_regions_base` (build an `EquilibriumRegion` with MRST's
  defaults - datum, contacts, reference phase, pc, endpoints),
  `get_min_max_phase_saturations` (`getMinMaxPhaseSaturations` - connate water
  floor, mobile range), and `init_state_equil_regions` (equilibrate a
  multi-region field, each region on its cell partition, scattered into the
  field arrays). `validate_phase353.py` proves it against the validated
  single-region path: the factory region equilibrates identically to a
  hand-built one (0.0), two regions with the same contact reproduce the
  single-region whole-field equilibrium (0.0), and two regions with different
  contacts give independent columns (one all-oil, one all-water). Phase 42
  still passes. **Residual follow-on:** the deck-driven `EQLNUM` region
  assignment and the compositional/relperm-endpoint region variants
  (`getInitializationRegionsDeck` / `...Compositional` /
  `getMinMaxPhaseSaturationsFromRelPerm`) land with AC17's deck path.

### AC17 - Deck and ECLIPSE initialization ~~[live-oil equilibration DONE - Phase 354; initEclipseProblemAD wrapper deferred]~~
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
- **Landed (Phase 354).** `init_state_deck` (explicit + equilibrated dispatch)
  was already present; closing AC17 meant making the **live-oil equilibration**
  path actually work and match MRST, which it never had - Phase 42 validated
  equilibration only for an analytic *dead-oil* fluid. Three real bugs surfaced
  and were fixed, all in the deck-fluid equilibration path:
  - `pvt_interp` crashed on a *scalar* pressure (0-d array), which the ODE
    integrator feeds it one depth at a time. Fixed to lift scalars to `(1,)` and
    return in the caller's shape (`ad_props/tables.py`).
  - `_phase_densities` handed `b_o` a `None` rs for a live fluid, producing
    **NaN** and sending the integrator into an unbounded step-halving loop (a
    300-cell case ran >300 s and never finished). Replaced with MRST's exact
    `getOilDensity`/`getGasDensity`: `rs = min(rs_given, rsSat)`, saturated flag
    `rs >= rsSat`, `rho_o = b_o(p, rs)(rhoS_o + rs rhoS_g)`, defaulting to
    saturated absent an RSVD `rs`. Now instant.
  - `initializeEquilibriumSaturations` dropped **connate water** (`s_min = 0`)
    and resolved no-oil cells to a degenerate `sw = sg = 0.5`. Ported MRST's
    two-step fill: `s_min`/`s_max` from the fluid's connate saturations, and the
    over-filled-cell branch that re-solves the outer two phases (water vs gas)
    directly - turning the 50/50 mush into a clean gas-over-water contact.
  - `initStateBlackOilAD`'s `rs` assignment now caps against `rsSat(p_oil)` (the
    oil-phase pressure, not the carried reference pressure, which differ below
    the water contact), and `EquilibriumRegion` gained `rs`/`rv` (constant or
    callable, RSVD-style).

  `validate_phase354.py` runs a live-oil column on SPE 1's real PVTO/PVDG tables
  with a constant `rs = rsSat(p_datum)` - undersaturated throughout, the exact
  branch that used to NaN - and matches MATLAB `initStateBlackOilAD` at machine
  precision: pressure 1.7e-15, saturations exactly 0.0, rs 7e-15.

  **Deferred:** the `initEclipseProblemAD` wrapper (it pulls in
  `convertDeckScheduleToMRST` = AC20 and full model construction), `getDeckOPMData`
  (OPM test-deck fetch), and `plotRegionContacts` (plotting).

---

## Tier 6 - adjoint simulators (MRST-layout homing)

### AC18 - Home the adjoint drivers under ad_core/simulators ~~[adjoint + perturbation homed DONE - Phase 360; parameter-sensitivity adjoint deferred]~~
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
- **DONE (Phase 360):** the adjoint drivers now have their MRST home under
  `ad_core/simulators/`. `compute_gradient_adjoint_ad.py` re-exports the backward
  sweep (`computeGradientAdjointAD`) from `optimization.adjoint` at MRST's path;
  `compute_gradient_perturbation_ad.py` is new - the finite-difference gradient
  (`computeGradientPerturbationAD`) that perturbs one control at a time and
  re-simulates, MRST's own ground-truth for an adjoint. Both sit side by side as
  in MRST's simulators directory and are imported from their submodule (kept out
  of the package `__init__` to avoid an `ad_core -> optimization -> ad_core`
  import cycle). `validate_phase360.py` runs the Phase 18 adjoint-vs-central-FD check
  through the new entry point (max rel err 8e-10) and the perturbation-vs-adjoint
  cross-check control for control (8e-10 central, 5e-6 forward) - no MATLAB, by
  the nature of the check: an adjoint is exact iff it matches a finite
  difference, and the FD is the independent ground truth. The perturbation driver
  uses a *relative* step (`perturbation * max(|u|, 1)`) so it is accurate across
  the rate/bhp scale gap out of the box; MRST's literal absolute-step
  forward-difference form is available via `central=False`.
- **Deferred:** `computeSensitivitiesAdjointAD` - the general
  *parameter*-sensitivity adjoint (dJ/d{transmissibility, pore volume,
  well index, ...}, not just controls). The control-objective adjoint is homed
  and validated; extending the backward sweep to model parameters is a genuine
  new capability (per-parameter residual derivatives), not relocation, and waits
  on a consumer - the history-matching path in `ensemble/` is the natural one.

---

## Tier 7 - utils (the genuine capability, not the plumbing)

### AC19 - Convergence ~~[DONE - Phase 355]~~
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
- **Landed (Phase 355).** `ad_core/utils/convergence.py` holds the kernels -
  `formation_volume_factor_avg` (`B_avg = mean(1/b)`), `cnv_mb_values` /
  `cnv_mb_convergence`, `well_control_residual`, `max_extra_residual`,
  `print_convergence_report` - and the **five** black-oil models
  (`generic`, `water`, `three_phase`, `two_phase_oil_water`, and through it
  `tracer`) now call them instead of each carrying its own copy of the formula.
  That copy was where the `B_avg` fix had to be applied five times before; there
  is now one place. `validate_phase355.py` follows the "fix the inputs, compare
  the algorithm" contract: it feeds the kernel MRST's *own* per-phase residuals
  (the two codes initialise the well bhp differently, so their iteration-1
  residuals are not shared - that is the well model's business, not the
  criterion's) and reproduces MRST's `getConvergenceValues` output number for
  number (CNV to ~1e-20, MB at machine zero), with `B_avg` and pore volume
  cross-checked as matching. Full sweep unchanged: 343 passing, 0 failing.
  **Deferred:** wiring these into a standalone `NonLinearSolver.checkConvergence`
  entry point (the models still call the kernels directly, which is enough for
  parity); the printed-report path exists but is unused.

### AC20 - Schedule tooling ~~[structured schedule + editing + solver coupling DONE - Phase 356; deck parsing deferred]~~
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
- **Landed (Phase 356).** `ad_core/utils/schedule.py` ports MRST's schedule
  structure - `Schedule` (`step_val` / `step_control` / `controls`), `Control`
  (`W`/`bc`/`src`) - and the editing operations: `simple_schedule`,
  `compress_schedule` (MRST's exact merge of consecutive same-control steps),
  `refine_schedule` (uniform subdivision under a max `dt`), `combine_schedules`,
  `subset_schedule`. **The capability that was missing** is time-varying
  controls: `NonLinearSolver.simulate_schedule` now accepts a `Schedule` and
  applies each control period's wells/bc/sources as time advances, so a well can
  tighten its bhp or step its rate partway through a run - which the old
  dt-array schedule (fixed wells for the whole run) could not express. The bare
  dt-array path is kept, so every existing caller is unchanged.

  `validate_phase356.py` validates it two ways, both MATLAB-free by design (the
  operations are structural and the coupling is behavioural): the round-trip
  invariants - total time preserved, `compress` leaves one step per control
  period, `compress(refine(s))` recovers the control history - and an end-to-end
  run where a producer's bhp drops 150 -> 120 bar at a control step, confirming
  the change takes effect (the bhp switches) and does physical work (the harder
  drawdown lifts more oil). Full sweep unchanged.

  **Deferred:** `convert_deck_schedule_to_mrst` - it needs a general deck
  SCHEDULE reader (WCONPROD/WCONINJE/TSTEP/DATES) and the spec's SPE9 deck,
  neither of which is present in this tree; and `make_schedule_consistent` /
  `convert_report_to_schedule`. A control step that changes the *number* of
  wells (opening or shutting one) is also refused for now, since the well state
  would need re-initialising - a clear error, not a silent shape mismatch.

### AC21 - CFL estimates and discretization setters ~~[saturation CFL DONE - Phase 357; composition CFL + setters deferred]~~
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
- **DONE (Phase 357):** `ad_core/utils/cfl.py` ports `estimateSaturationCFL`
  and the two-phase branch of `getFractionalFlowMagnitude`
  (`fractional_flow_derivative`): the Buckley-Leverett wave speed `|df_w/ds_w|`
  by seeding `sW` as the AD variable, then the per-cell CFL
  `(dt/pv)·|v_T|·|df/ds|` with upstream weighting, cross-flow handling, and the
  `useInflow` throughput option. Validated in `validate_phase357.py` against
  MRST's own `estimateSaturationCFL` on a 1D water/oil column with a fixed
  saturation front and interior flux - F and CFL match number for number
  (max|dF| 2e-16, max|dCFL| 2e-18), with N and pv confirmed identical.
- **Deferred:** the *three-phase* CFL (largest eigenvalue of the fractional-flow
  Jacobian) is flagged with a clear `NotImplementedError`; `estimateCompositionCFL`
  needs a compositional model (not yet ported); the capillary diffusion term
  (`Q` in `getFractionalFlowMagnitude`) is stubbed in MRST itself. The
  discretization setters (`setTimeDiscretization`/`setWENODiscretization`/
  `setMPFADiscretization`) wait on the WENO/MPFA discretizations they toggle.

### AC22 - Numeric utility functions ~~[core numeric helpers DONE - Phase 358; deck/N-d helpers deferred]~~
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
- **DONE (Phase 358):** ported the genuinely-numeric helpers still missing, one
  file each mirroring MRST's layout, all with self-tests:
  `phase_densities_to_bfactor.py`, `interpolate_idw.py`, `fast_interp_table.py`,
  `split_face_cell_value.py`, `merge_ordered_arrays.py`,
  `convert_reservoir_fluxes_to_surface.py`. The four with non-trivial arithmetic
  (IDW, table interp+extrapolation, dead/live b-factors, the supported
  `mergeOrderedArrays` insertion cases) are pinned to MRST in
  `validate_phase358.py` - all match to 0.0 (machine exact). Already present
  before this phase and not re-ported: `face_upstr` (in the upwind state
  function), `expand_matrix_to_cell` (compositional utils), `calculate_phase_rate`
  (facility state function).
- **MRST bug found:** `mergeOrderedArrays` is fragile on inputs its callers never
  send - its end-of-array shortcuts compare a pointer *index* against an element
  *value*, and its subset early-return calls `all(new == old)` on mismatched
  sizes. Both raise inside MATLAB (verified on `([1,2,3,4],[1,3,4])` and
  `([1,2,4],[1,3,4])`). The port reproduces the supported "insertion" case number
  for number and raises on the degenerate inputs rather than mis-ordering.
- **Deferred (as-needed):** `getMultipliers` (needs the deck rock-multiplier
  keywords MULTX/… which wait on the deck reader), `getMultiDimInterpolator`
  (N-dimensional gridded-interpolant closure, no consumer in the port yet),
  `expandIfUniform` (a scalar/vector broadcast convenience with no current
  caller). Port each when a consuming phase first needs it.

### AC23 - I/O, result handling, standalone solve, model selection ~~[factory + solve + well output DONE - Phase 359; ResultHandler + getReportOutput deferred]~~
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
- **DONE (Phase 359):** three files, each with self-tests:
  `select_model_from_deck.py` (the factory; the routing decision is split into a
  testable `deck_model_choice` so it can be checked without a full 3-phase
  fluid), `standalone_solve.py` (one-call single step onto the nonlinear
  solver's `solve_timestep`), `get_well_output.py` (the `(nsteps, nwells,
  nfields)` well time-series extractor, fields/wells by name or index). The
  factory's routing is pinned to MRST in `validate_phase359.py` on the real SPE1
  deck: MRST selects `GenericBlackOilModel` with disgas on / vapoil off, and the
  port reaches the identical decision - including SPE1's override branch, where
  BLACKOIL is off so disgas comes from the explicit RUNSPEC DISGAS, not the PVTO
  table. `get_reservoir_model` already exists (`wrapper_model.py`).
- **Deferred:** the compositional and EOR (polymer/surfactant) branches of the
  factory raise a clear `NotImplementedError` - their deck-construction path (an
  EOS model, the ad-eor models) is not ported, so returning a black-oil model
  would be wrong. `ResultHandler`/`HandleStruct` (MRST's on-disk state store,
  ~400 lines) and `getReportOutput` (report-tree walker) have no consumer in the
  port yet and are left for when one appears; `setReservoirModel` likewise.

---

## Tier 8 - backend performance internals

### AC24 - Diagonal Jacobian variants and operator helpers ~~[equivalence validated + numeric helpers DONE - Phase 361; block layouts deferred on the performance gate]~~
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
- **DONE (Phase 361):** `backends/diagonal/utils.py` ports the layout-independent
  numeric helpers with self-tests - `matrix_dims` (`matrixDims`),
  `get_sparse_arguments` (`getSparseArguments`, the offset `(I, J, V)` triplet
  for block assembly), and `diag_mult` (`diagMult`, diagonal row-scaling with
  MRST's all-zero / scalar / empty fast paths). `validate_phase361.py` is the
  equivalence+benchmark the tier calls for: a conservation-law residual
  (`div(upw(mob)*T*grad p)` + accumulation) assembled over a 2500-cell grid
  through both backends by feeding the *same* operators an ADI or a diagonal
  `GenericAD` variable - the assembled Jacobians are **identical to machine
  precision** (max|dJ| ~ 1.6e-19, same nnz), which transitively validates the
  diagonal backend since the sparse one is pinned to MATLAB elsewhere.
- **Finding (honest benchmark):** on the same case the pure-NumPy diagonal
  backend is **~90x *slower*** than the sparse backend, not faster. Its MRST
  speed-up lives in the MEX block kernels (out of scope), and the current
  diagonal `div` rebuilds its connectivity matrix each call. So the diagonal
  layout is *correct* here but not yet *fast* - reported, not hidden.
- **Deferred on the performance gate:** `FixedWidthJacobian` /
  `ConservationLawJacobian` / `getSparseBlocksFixedWidth` and `blockCSRtoSparse`
  (which untransposes MEX-specific block storage, meaningless without the MEX
  path). In pure NumPy these alternative layouts add complexity with no
  assembly-cost win - the tier's own gate ("port when assembly cost is the
  ceiling") is not met, and the benchmark above confirms the ceiling here is the
  operator overhead, not the Jacobian layout. Port them with the MEX kernels or a
  Numba block-assembly path, if either is ever in scope.

---

## Tier 9 - upscaling

### AC25 - Model, schedule, and state upscaling ~~[DONE - Phase 362]~~
- **What:** `upscaleModelTPFA`, `upscaleSchedule`, `upscaleState`.
- **Path:** `ad_core/upscale/`.
- **Why here:** coarsening an AD model + its schedule + its state onto a coarse
  grid, the AD-side companion to the flow-based upscaling PyMRST already has in
  the `upscaling` module. Independent of the rest.
- **Depends on:** `upscaling` (present), Tier 1.
- **Validation:** an upscaled SPE10 slab against MRST's coarse model response.
- **Size/risk:** M.
- **DONE (Phase 362):** three files under `ad_core/upscale/`, each with
  self-tests, built on the existing `upscaling` coarse-grid machinery:
  - `upscale_model_tpfa.py` (`upscaleModelTPFA`) - a runnable coarse model on the
    partition's coarse grid, coarse rock with pore-volume-conserving porosity
    (reusing `upscale_poro`) and MRST's default harmonic-mean permeability, coarse
    TPFA operators from that rock; carries the partition for the other two.
  - `upscale_state.py` (`upscaleState`) - pore-volume-weighted coarsening of
    pressure and saturations, with `rs`/`rv` weighted by oil/gas pore volume
    (`sO*pv`, `sG*pv`) and the no-phase block guarded to 0, not inf.
  - `upscale_schedule.py` (`upscaleSchedule`) - wells remapped to their unique
    coarse blocks, collapsed perforations' well indices combined (default `sum`,
    the conductance-preserving choice; `harmonic`/`mean` available).
  - `validate_phase362.py` pins the coarse rock and coarse state to MRST's own
    `upscaleModelTPFA`/`upscaleState` on a deterministic 6x6 -> 3x3 model:
    permeability and porosity match to **0.0**, saturation to 1e-16, pressure to
    1e-9 - machine exact, with the partition numbering confirmed identical first.
- **Difference from MRST, noted:** MRST's `upscaleSchedule` defaults to
  `recompute` (a fresh `computeWellIndex` on the coarse grid, needing a coarse
  well-index routine for an arbitrary perforation direction); this port defaults
  to `sum`, which is well-defined from the fine well index alone.
- **Deferred:** the compositional branch of `upscaleState` (EOS mole-fraction
  coarsening + re-flash) and coarse-face flux coarsening (needs the coarse grid's
  fine-face connectivity), plus `upscaleSchedule`'s bc/src coarsening - none has a
  consumer in the port yet. The SPE10-slab end-to-end coarse *run* comparison
  waits on SPE10 deck data; the per-routine MATLAB checks above cover the math.

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
  a good validation case (as several already have). **Ported so far:**
  `adBuckleyLeverett1D` -> `ad_core/examples/ad_buckley_leverett_1d.py` (Phase
  363): the incompressible 1D waterflood, flux BC in / pressure BC out, 1 PVI in
  25 steps; `validate_phase363.py` matches MRST's full 50-cell saturation profile
  to 3e-7 (solver tolerance), inlet/outlet/breakthrough included. The script's
  plotToolbar / general-solver-hook / `ResultHandler` tails are presentation and
  IO layers, dropped. Still blocked, each on a listed gap: `MPFAvsTPFAwithADExample`
  (MPFA), `wenoExampleAD` (WENO, deferred in AC21), `linearSolverExample`
  (AGMG/external solvers), `blockAssemblyBigModelExample` + the book-ii backend
  benchmarks (MEX/block assembly), `demoPackedProblems` /
  `ensemblePackedProblemsExample` (packed-problem infra), `fluidInspectionExample`
  / `mrst_demo` (GUI), `compositeModelTutorial` (SPE10 data + composite models).
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
