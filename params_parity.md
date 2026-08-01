# Params parity ledger

## Scope

This ledger compares:

```text
mrst-2024a-Geomech/params/
PyMRST/pymrst/params/
```

The MRST tree contains 21 files: 15 executable functions, 3 catalogs, and 3
README files. Every executable function has a Python owner. Port status and
validation depth are tracked separately.

## Summary

| Classification | Count |
|---|---:|
| Executable MRST functions | 15 |
| Direct Python owners | 15 |
| Baseline MATLAB-validated | 15 |
| Behaviorally complete at full option scope | 15 |
| Broadly implemented but requiring edge/option completion | 0 |
| Explicit partial implementation | 0 |
| Catalog files excluded | 3 |
| Documentation files requiring Python equivalents | 0 |

“Baseline MATLAB-validated” means at least one representative numerical path
matches MRST. It does not imply complete option or edge-case parity.

## Status definitions

- **Complete**: public behavior and material options are implemented and
  validated across representative edge cases.
- **Broad**: primary numerical behavior works and has MATLAB evidence, but
  advanced options or edge semantics remain.
- **Partial**: a documented MRST numerical branch is refused or absent.
- **Excluded catalog**: MATLAB package index, not executable behavior.
- **Documentation covered**: source documentation has a dedicated Python
  counterpart in the equivalent package location.

## Root

| MRST file | Python owner | Port status | Validation | Remaining work |
|---|---|---|---|---|
| `gravity.m` | `pymrst/params/gravity_const.py`, `gravity/` | Complete | `validate_params_gravity.py`, 7 focused tests, `validate_phase2.py` | Stateful facade matches MRST; solver APIs intentionally retain explicit-vector priority |
| `Contents.m` | `pymrst/params/__init__.py` | Excluded catalog | Import tests | Keep public export inventory current |
| `README` | `pymrst/params/README.md` | Documentation covered | Review and audit | Keep package overview and links current |

## Rock and petrophysics

| MRST file | Python owner | Port status | Validation | Remaining work |
|---|---|---|---|---|
| `rock/makeRock.m` | `rock/make_rock.py` | Complete | `validate_params_rock_tensor.py`, 16 shared focused tests, `validate_phase2.py` | Every compact layout, per-cell properties, NTG, warnings and shapes covered |
| `rock/permTensor.m` | `rock/perm_tensor.py` | Complete | `validate_params_rock_tensor.py`, focused tests | Full 1-D/2-D/3-D expansion values and index arrays covered |
| `rock/poreVolume.m` | `rock/pore_volume.py` | Complete | `validate_phase2.py` | Retain volume/porosity/NTG regression coverage |
| `rock/gaussianField.m` | `rock/gaussian_field.py` | Complete | `validate_params_stochastic.py`, focused tests, `validate_phase22.py` | 1-D/2-D/3-D kernels, validation, constants and shared-draw parity covered |
| `rock/logNormLayers.m` | `rock/log_norm_layers.py` | Complete | `validate_params_stochastic.py`, focused tests, `validate_phase22.py` | Default/scalar/vector layers, uneven bounds, means, draws and errors covered |
| `rock/grdecl2Rock.m` | `rock/grdecl2rock.py` | Complete | expanded `validate_phase366.py`, 8 focused tests | Aliases, precedence, active maps, properties, multipliers, faults and malformed inputs covered |
| `rock/permeabilityConverter.m` | `rock/permeability_converter.py` | Complete | `validate_phase386.py`, 3 focused tests | Strict mode matches MRST's off-diagonal failure; explicit corrected mode supplies the usable symmetric transform |
| `rock/Contents.m` | `rock/__init__.py` | Excluded catalog | Import tests | Keep exports aligned |
| `rock/README` | `rock/README.md` | Documentation covered | Review and audit | Keep API examples current |

## Wells and boundary conditions

| MRST file | Python owner | Port status | Validation | Remaining work |
|---|---|---|---|---|
| `wells_and_bc/addBC.m` | `wells_and_bc/add_bc.py` | Complete | `validate_params_forcing.py`, focused tests, `validate_phase2.py` | Per-face types, warning/no-op, broadcasting and saturation compatibility covered |
| `wells_and_bc/addSource.m` | `wells_and_bc/add_source.py` | Complete | `validate_params_forcing.py`, focused tests, `validate_phase2.py` | Empty default saturation, broadcasting, append and conflict semantics covered |
| `wells_and_bc/addWell.m` | `wells_and_bc/add_well.py` | Complete | `validate_params_end_to_end.py`, `validate_params_add_well.py`, focused tests, `validate_phase2.py` | Cartesian/corner-point, inactive, mixed/slanted, structure fields and all inner products covered |
| `wells_and_bc/pside.m` | `wells_and_bc/pside.py` | Complete | expanded `validate_phase22.py`, focused tests | Aliases, in-plane/perpendicular ranges, append, empty and errors covered |
| `wells_and_bc/psideh.m` | `wells_and_bc/psideh.py` | Complete | `validate_phase227.py` | Retain fluid-property and range regressions |
| `wells_and_bc/fluxside.m` | `wells_and_bc/fluxside.py` | Complete | expanded `validate_phase22.py`, focused tests | Scalar/vector area weighting, ranges, append, empty and errors covered |
| `wells_and_bc/verticalWell.m` | `wells_and_bc/vertical_well.py` | Complete | `validate_phase22.py`, 5 focused mode tests, Norne workflows | Python exposes MRST's ambiguous second signature explicitly as `top_cell=True` |
| `wells_and_bc/Contents.m` | `wells_and_bc/__init__.py` | Excluded catalog | Import tests | Keep exports aligned |
| `wells_and_bc/README` | `wells_and_bc/README.md` | Documentation covered | Review and audit | Keep indexing and shape conventions current |

## PyMRST extensions

These files are useful PyMRST additions, not missing MRST counterparts:

| Python file | Purpose |
|---|---|
| `rock/rock.py` | Typed `Rock` data container |
| `wells_and_bc/types.py` | Typed `Well`, `BoundaryCondition`, and `Source` containers |
| `wells_and_bc/incomp_well_pressure_drop.py` | Connection hydrostatic pressure-drop helper |
| `wells_and_bc/multisegment.py` | Explicit multisegment-well topology structures |
| `gravity_const.py` | Explicit-vector gravity API used by Python solvers |

`computeWellIndex` is intentionally owned by
`pymrst/utils/compute_well_index.py`, matching its MRST 2024a location under
`utils/`, even though `addWell` consumes it.

## Highest-priority parity gaps

No executable parity gaps remain. Future work is maintenance: run the audit and
MATLAB fixture suite whenever params behavior or shared grid/solver code changes.

## Existing validation status

At the time this ledger was written:

```text
validate_phase2.py   -> all compared arrays match MATLAB
validate_phase22.py  -> all core helper comparisons match MATLAB
validate_params_add_well.py -> advanced options match MATLAB
validate_params_gravity.py  -> command state matches MATLAB
validate_params_forcing.py  -> BC/source structures match MATLAB
```

Additional focused validators exist for `psideh`, `grdecl2Rock`, and
`permeabilityConverter`. These are evidence for their covered branches, not a
claim of full option parity.

## Completion gate

The ledger can declare full parity only when:

- every executable row is `Complete`;
- no numerical branch is refused without an explicit accepted exclusion;
- every public option family has focused Python tests;
- every material numerical branch has MATLAB reference evidence;
- the end-to-end deck/rock/forcing/well workflow matches MRST;
- the parity audit reports zero partial and zero unvalidated executable rows.
