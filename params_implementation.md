# Params end-to-end implementation plan

## Objective

Complete `pymrst/params` against MRST 2024a's `params/` tree at API,
behavioral, numerical, layout, and validation parity.

The reference tree is:

```text
mrst-2024a-Geomech/params/
```

The implementation tree is:

```text
PyMRST/pymrst/params/
```

File presence alone is not the completion criterion. A function is complete
only when its supported inputs, optional arguments, returned fields, failure
semantics, indexing convention, and numerical outputs are tested and compared
with MATLAB/MRST where appropriate.

## Current baseline

MRST contains 21 files under `params/`:

- 15 executable MATLAB functions;
- 3 `Contents.m` catalogs;
- 3 README files.

All 15 executable functions have Python owners. Existing MATLAB validators
already establish important baseline parity:

- `validate_phase2.py`: `makeRock`, `permTensor`, `poreVolume`, `addBC`,
  `addSource`, `addWell`, and gravity reference cases;
- `validate_phase22.py`: `gaussianField`, `logNormLayers`, `pside`,
  `fluxside`, and `verticalWell` reference cases;
- `validate_phase227.py`: hydrostatic `psideh` behavior;
- `validate_phase366.py`: `grdecl2Rock` keyword conversion;
- `validate_phase386.py`: diagonal `permeabilityConverter` transforms.

The current baseline validators pass. The remaining work is advanced option,
edge-case, structure-field, and workflow parity, concentrated in `addWell`.

## Definition of done

`params` is complete only when:

1. Every MRST file is classified in `params_parity.md`.
2. Every executable function has a direct Python owner and public export.
3. Every supported MRST option is implemented or explicitly documented as a
   deliberate incompatibility.
4. Python uses zero-based cell/face indices consistently at public APIs.
5. Returned rock, well, BC, and source structures contain all fields needed by
   downstream MRST-equivalent solvers.
6. Empty inputs, logical masks, scalar expansion, per-entity arrays, aliases,
   and malformed inputs follow documented behavior.
7. Focused tests live under `tests/params/` rather than only in module
   `__main__` blocks.
8. MATLAB fixtures cover every major option family and numerical branch.
9. A realistic deck-to-rock-to-wells-to-forces workflow runs in both MRST and
   PyMRST and produces matching structures.
10. The parity ledger contains no partial or unvalidated executable entries.

## Compatibility conventions

### Indexing

MRST uses one-based indices; PyMRST public APIs use zero-based indices.
MATLAB fixture validators must convert exactly once at the language boundary.

### Array ordering

Deck arrays and logical Cartesian indices retain ECLIPSE/MRST Fortran order.
Operations involving `cartDims`, active mappings, and GRDECL properties must
use `order="F"` where required.

### Scalar expansion

Options accepted as either one value or one value per face, cell, or
perforation must be normalized centrally and validated before structures are
mutated.

### Mutation

MRST appends to structures. PyMRST may mutate dataclasses for compatibility,
but must not leave a partially updated object after validation fails.

### Numerical tolerance

- indices, flags, names, types, and shapes: exact;
- rock and forcing values: approximately `1e-12` relative where algorithms
  are identical;
- well indices: approximately `1e-10` relative;
- stochastic fields: compare exact outputs only when both languages receive
  the same random draw; otherwise compare declared statistical invariants.

## Phase 1: authoritative audit and tests

Create:

```text
scripts/audit_params_parity.py
docs/params/parity.csv
tests/params/
```

The audit should inventory all 21 MRST files, classify catalogs and README
files explicitly, track implementation and validation independently, and fail
if an executable entry becomes partial or loses its evidence.

Move reusable module self-tests into focused pytest files. Keep module smoke
tests only as developer conveniences.

## Phase 2: complete `addWell`

This is the largest and highest-priority batch.

Implement:

- numeric indices and full-length logical completion masks;
- MRST's empty-completion warning/no-op behavior;
- per-completion or scalar `Radius`, `Skin`, `Kh`, and `WI`;
- partially supplied `WI`, computing only negative entries;
- `InnerProduct` routing;
- complete `Comp_i`/`compi` alias and conflict behavior;
- `status` and default per-completion `cstatus`;
- `lims`, `vfp_index`, `cp`, `components`, and `heel` fields;
- `calcReprRad` and representative radius `rR`;
- `cellDims` override;
- `lineSegments` and projected directional well indices;
- exact reference-depth and gravity behavior;
- exact type/sign/control validation;
- complete returned `Well` structure fields.

Validation must cover vertical, horizontal, slanted, inactive-cell,
partially supplied WI, shut-well, shut-completion, representative-radius,
component, VFP, and line-segment cases.

### Phase 2 implementation progress

Implemented and validated on 2026-08-01:

- logical masks and warning/no-op behavior for empty selections;
- scalar and per-completion radius, skin, Kh, and WI normalization;
- partial WI calculation without overwriting supplied positive entries;
- `ip_tpf`, `ip_quasitpf`, `ip_simple`, `ip_rt`, and `ip_quasirt` well
  constants, including MRST's interpolated mimetic table;
- mixed completion directions, full-grid or completion-sized `cellDims`, and
  projected `lineSegments` indices;
- `status`, `cstatus`, `lims`, `vfp_index`, `cp`, `components`, `heel`,
  `defaulted`, and representative-radius `rR` structure fields;
- `calcReprRad=false`, reference-depth behavior, and rate/sign resolution;
- preservation of the legacy positional `Well` constructor layout.

Evidence:

```text
pytest -q tests/params/test_add_well_options.py       -> 7 passed
validate_params_add_well.py                          -> matches MATLAB MRST
validate_phase2.py                                   -> matches MATLAB MRST
```

The remaining Phase 2 acceptance work is an expanded corner-point and
inactive-cell fixture plus documenting Python's direct `compi` spelling for
MATLAB's legacy `Comp_i` alias.

That final acceptance work is now complete. The end-to-end fixture uses a
faulted corner-point GRDECL with an inactive completion, a mixed-direction
well, an inactive-filtered vertical well, gravity, BCs, a source, rock
properties, transmissibility, pore volume, initialization, and a complete
incompressible pressure solve. Geometry, WI, `dZ`, `rR`, pressure, divergence,
connection fluxes, and BHP all match MATLAB. Python uses `compi`; MATLAB's
legacy spelling is `Comp_i`. A MATLAB row-character `Dir='xyzx'` triggers an
MRST implicit-expansion bug in `rR`; the fixture uses the documented
per-completion character column, while PyMRST always preserves one radius per
completion.

## Phase 3: complete `verticalWell`

Implement both MRST call modes:

1. logical `(i, j, layers)` columns;
2. a top-cell index plus layers.

Also support:

- Cartesian and corner-point `cartDims` grids;
- layered grids with `layerSize` and `numLayers`;
- empty layer lists meaning all layers;
- inactive-cell mapping;
- zero active completions following MRST behavior;
- forced `z` direction regardless of caller input;
- complete passthrough of all `addWell` options.

### Phase 3 implementation progress

Implemented on 2026-08-01. `vertical_well` now supports logical `(i, j)`
columns, an explicit zero-based `top_cell=True` mode, Cartesian and layered
grids, empty layer selections as all layers, inactive completion filtering,
empty-column no-op behavior, forced vertical direction, and passthrough of
the expanded `add_well` surface. The explicit flag avoids the positional
ambiguity that Python would otherwise have between MRST's two signatures.

Evidence:

```text
pytest -q tests/params/test_vertical_well_modes.py -> 5 passed
validate_phase22.py                               -> matches MATLAB MRST
```

## Phase 4: gravity compatibility facade

Retain explicit solver gravity vectors, but add an MRST-compatible state
manager supporting:

- `on`, `off`, and `reset`;
- `x`, `y`, and `z` directions;
- boolean enable/disable;
- scalar acceleration strength;
- explicit two- and three-component vectors;
- querying the current vector;
- isolated reset behavior in tests.

Downstream code should accept an explicit vector first and use global
compatibility state only when explicitly requested.

### Phase 4 implementation progress

Implemented and cross-validated on 2026-08-01. The new process-local
`gravity()` facade supports ordered command combinations, `on/off/reset`,
all MRST string aliases, axis changes, booleans, scalar strength, explicit
two- and three-component vectors, and querying. `isolated_gravity()` provides
exception-safe state restoration for tests and scoped workflows. Existing
solvers retain their explicit gravity-vector interface.

```text
pytest -q tests/params/test_gravity_state.py -> 7 passed
validate_params_gravity.py                  -> 11 transitions match MRST
validate_phase2.py                          -> solver cases still match MRST
```

## Phase 5: boundary conditions and sources

### `addBC`

Complete:

- a single type or one type per face;
- empty face lists returning the original structure;
- scalar/per-face values;
- scalar, row, and per-face saturation arrays;
- saturation-column compatibility across append calls;
- duplicate-face rejection without partial mutation;
- exact case-insensitive type normalization.

### `addSource`

Complete:

- MRST's empty default saturation rather than an implicit value of one;
- empty source lists;
- scalar/per-cell rates;
- row/per-cell saturation broadcasting;
- saturation-column compatibility across calls;
- duplicate-cell rejection;
- PyMRST component extensions without changing MRST base-field semantics.

### Side helpers

Expand `pside`, `psideh`, and `fluxside` tests across every side, logical
range form, empty selection, saturation shape, existing-structure append, and
hydrostatic density/mobility combination.

### Phase 5 implementation progress

The base `addBC` and `addSource` parity work was completed on 2026-08-01:
per-face BC types, empty BC warning/no-op, scalar and row broadcasting,
column compatibility, validation-before-mutation, MRST's empty default source
saturation, source append semantics, and aligned PyMRST component extensions.

```text
pytest -q tests/params/test_add_bc_source_options.py -> 7 passed
validate_params_forcing.py                          -> matches MATLAB MRST
validate_phase2.py                                  -> matches MATLAB MRST
```

The side-helper batch is now complete as well. `boundary_face_indices`,
`pside`, `fluxside`, and `psideh` support empty in-plane ranges, validated
zero-based ranges, perpendicular search ranges, dimensional side checks,
aliases, empty-selection warning/no-op behavior, append behavior, and scalar
or vector flux area weighting. The vector-flux branch previously skipped
MRST's area weighting and was corrected.

```text
pytest -q tests/params/test_side_helper_options.py -> 6 passed
validate_phase22.py -> vector flux and perpendicular range match MRST
validate_phase227.py -> hydrostatic side pressure matches MRST
```

## Phase 6: rock and petrophysics

### `makeRock`, `permTensor`, and `poreVolume`

Cover scalar, isotropic, diagonal, and full symmetric tensors; cell subsets;
2-D/3-D shapes; NTG scaling; missing porosity; and invalid dimensions.

`makeRock` and `permTensor` are now complete across every MRST compact tensor
layout in 1-D, 2-D, and 3-D, including full expansion indices and deterministic
shape errors. In Python, a one-dimensional array containing `nc` entries is a
per-cell scalar field; callers use a two-dimensional single row for a uniform
anisotropic tensor. This resolves NumPy's lack of MATLAB row/column-vector
distinction without sacrificing either input form.

```text
pytest -q tests/params/test_make_rock_perm_tensor.py -> 16 passed
validate_params_rock_tensor.py                     -> matches MATLAB MRST
validate_phase2.py                                 -> solver cases unchanged
```

### `grdecl2Rock`

Validate:

- `PERMX/Y/Z`, `PERMH`, and `PERMXX/YY/ZZ`;
- all symmetric off-diagonal aliases;
- isotropic collapse and diagonal/full tensor layouts;
- `PORO`, `NTG`, and `ROCKTYPE` combinations;
- active maps supplied as indices or grids;
- multiplier fields and fault metadata;
- inconsistent tensors and malformed keyword lengths.

This matrix is now implemented and cross-validated. Coverage includes lone
isotropic keywords, diagonal and full tensors, `PERMH`, long diagonal names,
forward/reverse off-diagonal aliases and their MRST precedence, numeric/grid
and logical active maps, `PORO`, `NTG`, `ROCKTYPE`, positive/negative-direction
multipliers, fault metadata, and deterministic validation errors. A bug that
collapsed `MULTX` and `MULTX_` into one field was corrected; they now map to
`multipliers.x` and `multipliers.x_` respectively.

```text
pytest -q tests/params/test_grdecl2rock_options.py -> 8 passed
validate_phase366.py                              -> all branches match MRST
```

### `permeabilityConverter`

Complete tilted, skewed, bedding-plane, and bedding-plane-normal transforms.
Resolve the off-diagonal branch explicitly: either reproduce MRST 2024a's
observable behavior in strict-compatibility mode or provide a corrected
tensor transform with the difference documented and independently validated.

This policy is now implemented. Strict `off_diagonal_mode="mrst"` records and
reproduces MRST 2024a's observable refusal (the MATLAB implementation fails in
its cell-array `switch`). Explicit `off_diagonal_mode="corrected"` performs the
full symmetric cross-axis tensor transform using the supplied `PERMXY`,
`PERMYZ`, and `PERMZX` values. Diagonal tilted-grid results remain numerically
identical to MRST.

```text
pytest -q tests/params/test_permeability_converter_modes.py -> 3 passed
validate_phase386.py -> diagonal tensors match; MRST refusal confirmed;
                        corrected PyMRST tensors finite
```

## Phase 7: stochastic rock fields

Complete dimensional and validation behavior for `gaussianField` and
`logNormLayers`:

- scalar/vector kernel sizes;
- 1-D, 2-D, and 3-D fields where MRST permits them;
- degenerate constant inputs;
- explicit layer boundaries and uneven groups;
- scalar and per-layer target means;
- invalid partitions and kernel parameters;
- deterministic fixtures using shared random draws;
- distribution, mean, range, and correlation invariants for native RNG runs.

NumPy and MATLAB seed equality is not a parity requirement because their RNG
streams differ. Deterministic cross-language comparisons must inject the same
draws.

This phase is complete. `gaussian_field` now validates 1-D/2-D/3-D sizes,
scalar or dimensional kernels, odd positive masks, intervals, standard
deviation, and injected-field shapes. `log_norm_layers` supports MRST's
default per-model-layer mode, scalar random layer counts, vector target means,
explicit uneven boundaries, 2-D convenience input, returned zero-based layer
boundaries, injected draws/shifts, and complete validation. Constant Gaussian
draws follow MRST's undefined zero-range rescaling instead of amplifying FFT
roundoff into a false heterogeneous field.

```text
pytest -q tests/params/test_stochastic_rock_fields.py -> 20 passed
validate_params_stochastic.py -> deterministic 3-D Gaussian and uneven
                                 lognormal layers match MATLAB MRST
validate_phase22.py            -> existing shared-draw cases still match
```

## Phase 8: end-to-end MATLAB validation

Add a paired workflow that:

1. processes a GRDECL with active/inactive cells and tensor permeability;
2. constructs `Rock` using `grdecl2Rock`;
3. creates pressure and flux boundary conditions;
4. creates sources;
5. adds vertical, horizontal, and line-segment wells;
6. applies gravity and reference depths;
7. serializes every resulting field from MATLAB;
8. compares Python structures, indices, values, and shapes;
9. runs a downstream transmissibility and initialization calculation.

Completed by `dump_params_end_to_end_reference.m` and
`validate_params_end_to_end.py`. The validation goes beyond initialization and
compares a solved pressure field using topology-independent physical
invariants, so equivalent corner-point face numbering does not create false
failures.

```text
validate_params_end_to_end.py -> all 28 comparisons match MATLAB MRST
```

## Final acceptance commands

The final gate should include:

```bash
python3 scripts/audit_params_parity.py
pytest -q tests/params
python3 examples/validation/validate_phase2.py
python3 examples/validation/validate_phase22.py
python3 examples/validation/validate_phase227.py
python3 examples/validation/validate_phase366.py
python3 examples/validation/validate_phase386.py
python3 examples/validation/validate_params_add_well.py
python3 examples/validation/validate_params_gravity.py
python3 examples/validation/validate_params_forcing.py
python3 examples/validation/validate_params_end_to_end.py
```

Completion requires zero pending executable files, zero partial functions,
all focused tests passing, and all MATLAB comparisons passing.

## Integration verification

The completed params layer has also been exercised through its downstream
consumers:

```text
gridprocessing, utils, deckformat, ad_props, plotting -> 587 passed, 1 skipped
vem, vemmech, ad_blackoil                         -> 192 passed, 1 skipped
ad_core                                           -> 201 passed
```

The repository-wide `tests/test_all.py` harness launches every module as a
separate subprocess with long per-module timeouts. A diagnostic run was
stopped after 839 passing tests and one skip, with no assertion failure; the
targeted downstream groups above were then run to completion.
