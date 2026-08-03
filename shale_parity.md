# PyMRST shale parity with MRST 2024a

## Audit status

**Current conclusion: the shale module is not implemented end-to-end.**

PyMRST now has native shale data, utility, NNC, pEDFM, well and compositional
owners and imports its fracture framework through the consolidated `hfm` owner. All six distributed
data assets are packaged. The projected-NNC owner matches a direct centered
axis-aligned MRST case and competitive Langmuir sorption is inserted into the
actual natural-variable component residuals with AD Jacobians. Every executable
MRST owner now has a native Python owner and focused coverage. Remaining work is
certification depth: direct equation/operator fixtures, complete example time
histories, the unavailable ADFNE Bakken realization, and field-scale regression
records. The compositional owner covers both MRST catalog variants directly.

Audit reference: `mrst-2024a-Geomech/modules/shale`.

## Status definitions

| Status | Meaning |
|---|---|
| Validated | Native owner implemented, focused tests pass, and MATLAB parity evidence exists |
| Implemented | Native owner and focused Python tests exist, but MATLAB certification is incomplete |
| Partial | Some behavior exists in the correct owner; material reference behavior is absent |
| Reusable dependency | Related lower-level functionality exists elsewhere, but the shale owner is absent |
| Surface only | Import/name exists but redirects to different or narrower functionality |
| Data missing | Required reference asset is not packaged |
| Missing | No native Python owner |
| Catalog | Documentation/catalog file, not a numerical owner |

Filename similarity alone is not parity. A shared HFM implementation is marked
as a reusable dependency until the shale-specific data contract and MATLAB
behavior are demonstrated.

## Inventory summary

| Reference group | MATLAB owners | Data/docs | Current parity |
|---|---:|---:|---|
| `compositionalFns` | 4 | 0 | 3/4 implemented, 1/4 validated |
| `hfmFns` | 5 | 0 | 2/5 implemented, 3/5 validated |
| `otherMRSTfns` | 4 | 0 | 1/4 implemented, 3/4 validated |
| `pEDFM_nnc` | 3 | 0 | 1/3 implemented, 2/3 validated |
| `utils` | 8 | 0 | 3/8 implemented, 4/8 validated |
| `examples` | 13 executable | 1 catalog + 5 MAT files | 13/13 implemented; all MAT data packaged |
| `SPE201243PA` | 10 | 1 CSV | 6/10 implemented, 4/10 validated; CSV packaged |
| Root | 0 | 2 documents | README/reference notes absent |
| **Total** | **47 executable owners** | **9 data/catalog/docs files** | **30 implemented, 0 partial, 17 validated shale owners; all executable owners present** |

The current import façade is tracked separately and is not part of the MRST
owner count.

## Current PyMRST surface

| PyMRST symbol | Current behavior | Status | Reason |
|---|---|---|---|
| `EDFMGrid`, `FractureLine`, `PlanarFracture` | Imported by shale through `pymrst.modules.hfm` | Reusable dependency | Consolidated public dependency path; they are not shale owners |
| `build_shale_edfm` | Calls generic 2D `build_edfm` | Surface only | MRST shale is a 3D compositional EDFM/pEDFM module |
| `build_shale_edfm_3d` | Calls generic `build_edfm_3d` | Surface only | Lower-level convenience builder; native shale preprocessing is provided by `edfm_shale_grid` |
| `solve_incomp_edfm*` | Re-exported generic incompressible solvers | Reusable dependency | Reference workflows require compressible compositional shale physics |
| `validate_phase106.py` | Checks import surface and generic HFM propagation | Surface only | No shale owner, physics, data, MATLAB or paper-case validation |

## Owner-by-owner parity ledger

### `compositionalFns`

| MRST owner | Required Python owner | Status | Missing behavior/evidence |
|---|---|---|---|
| `NatVarsShaleModel.m` | `compositional_fns/nat_vars_shale_model.py` | Implemented | Native subclass dispatches shale equations; sorption accumulation, diffusion flux, Gangi matrix scaling and fracture/everywhere non-Darcy phase velocities are connected through AD-aware assembly hooks; MATLAB regression pending |
| `SorbedCompositionalFluid.m` | `compositional_fns/sorbed_compositional_fluid.py` | Implemented | Native table mixture and isotherm validation exist; direct MATLAB fixture pending |
| `eqnsShaleNaturalVars.m` | `compositional_fns/equations_shale_natural_vars.py` | Implemented | Competitive Langmuir accumulation is inserted componentwise; diffusion modifies component flux and Gangi/non-Darcy modify phase velocities before mass flux assembly. Kernel, masking and AD propagation tests pass; direct MATLAB residual/Jacobian fixtures pending |
| `getShaleCompFluidCase.m` | `compositional_fns/get_shale_comp_fluid_case.py` | Validated | All 18 main and seven utility-only cases match a direct MRST fixture for component ordering, thermodynamic arrays, streams, pressure/temperature, BIC and isotherms. PyMRST intentionally trims MRST's malformed eight-entry Eagle Ford injection stream to its six physical components; diffusion/Parachor are source-certified Python-only metadata because MRST does not return them |

PyMRST's compositional module is a substantial reusable dependency, but none of
these shale extensions currently exists.

### `hfmFns`

| MRST owner | Required Python owner | Status | Missing behavior/evidence |
|---|---|---|---|
| `EDFMshalegrid.m` | `hfm_fns/edfm_shale_grid.py` | Validated | Native convex polygon preprocessing clips against actual cell face-edge topology for Cartesian, oblique and tetrahedral/polyhedral cells; a direct MRST pipeline fixture matches matrix/global cell counts, fracture topology and NNC pairs, with broader topology fixtures still desirable |
| `assembleShaleGlobalGrid.m` | `hfm_fns/assemble_shale_global_grid.py` | Implemented | Appends fracture nodes, deduplicated shared edge-faces, neighbors, cell-face topology and geometry to a checked global `Grid`; exposes fracture cell/face/node maps and global rock/NNC data. Direct MATLAB topology fixture pending |
| `fracturefractureShaleNNCs3D.m` | `hfm_fns/fracture_fracture_shale_nncs_3d.py` | Implemented | Finite convex-plane intersections are segmented across every overlapping fracture-cell pair; line coverage is conserved and unequal aperture/permeability uses two-sided Moinfar resistance. Direct MATLAB fixture and intersection metadata fixture pending |
| `fracturematrixShaleNNC3D.m` | `hfm_fns/fracture_matrix_shale_nncs_3d.py` | Validated | Interior connections receive the MRST CI surface factor and Tene `adjustT`; direct MRST NNC pairs and transmissibilities match numerical precision. Face-aligned and incremental cases remain Python-covered |
| `setupShaleEDFMOpsTPFA.m` | `hfm_fns/setup_shale_edfm_operators_tpfa.py` | Validated | Direct MRST fixture matches all 75 active neighbor/transmissibility pairs and global pore volumes, including the 12 previously omitted within-fracture connections; AD closures remain focused Python-tested |

Generic 2D/3D construction is reachable through `hfm` and may provide
primitives. It has not been shown to reproduce these owners' structures or
MATLAB results. Missing MRST `hfm_v2` breadth remains an HFM parity gap and is
not implied by consolidating the former forwarding package.

### `otherMRSTfns`

| MRST owner | Required Python owner | Status | Missing behavior/evidence |
|---|---|---|---|
| `addWellShale.m` | `other_mrst_fns/add_well_shale.py` | Validated | Cell vectors, logical masks, matrix/fracture/mixed completions, directional and partially supplied WI, Kh/cell-dimension overrides, controls, limits and composition are Python-tested; a direct MRST fracture-completion fixture certifies cell mapping, WI, representative radius, depth, sign correction, VFP index and legacy payload |
| `computeWIShale.m` | `other_mrst_fns/compute_wi_shale.py` | Validated | Rectangular equivalent radius and WI match direct MRST fixture; hydraulic/NF completion selection tested |
| `makeShaleRock.m` | `other_mrst_fns/make_shale_rock.py` | Validated | Tensor expansion, porosity, NTG and matrix flags match direct MRST 2024a fixture |
| `wellfractureNNCs3D.m` | `other_mrst_fns/well_fracture_nncs_3d.py` | Validated | Segment/finite-plane intersection point, plane and global fracture cell match direct MRST fixture; multiwell edge cases tested |

### `pEDFM_nnc`

| MRST owner | Required Python owner | Status | Missing behavior/evidence |
|---|---|---|---|
| `pMatFracNNCs3D.m` | `pedfm_nnc/projected_matrix_fracture_nncs_3d.py` | Validated | Centered topology/area/direction/transmissibility match direct MRST. Exact convex polygon union handles overlap, containment and disjoint projections; affine-skew six-face cells and MRST `defaultFace` option validation are tested |
| `setupPEDFMOpsTPFA.m` | `pedfm_nnc/setup_pedfm_operators_tpfa.py` | Implemented | Direct MRST pipeline matches the 96-edge topology, projected MF/MM rows, projected areas and global pore volumes. The MRST 2024a fixture stores projected `G.nnc.T` on inconsistent rows (including zero T for positive area and nonzero T for zero area); PyMRST intentionally preserves the physical row invariant, producing a documented 2.7% sorted-vector difference instead of copying the reference defect |
| `transmultpEDFM.m` | `pedfm_nnc/transmissibility_multiplier_pedfm.py` | Validated | Every active grid-face transmissibility matches the direct MRST operator-pipeline fixture; orientation independence, duplicate-area reduction and clamping also tested |

This group is the largest parity blocker. Generic EDFM NNCs are not a substitute
for projection-based EDFM.

### `utils`

| MRST owner | Required Python owner | Status | Missing behavior/evidence |
|---|---|---|---|
| `Pts2CellIdx.m` | `utils/points_to_cell_indices.py` | Validated | Zero-based tensor-grid results match direct MRST 2024a fixture after index conversion |
| `createMultiStageHFs.m` | `utils/create_multistage_hfs.py` | Validated | Fracture points and centroid progression match direct MRST 2024a fixture |
| `gangiFn.m` | `utils/gangi.py` | Validated | Values match direct MRST 2024a fixture; explicit AD Jacobian fixture remains desirable |
| `getShaleCompFluidCase.m` | `utils/get_shale_comp_fluid_case.py` | Validated | Native extended owner adds the seven oil, SlotDrill and modified-Barnett cases to the 18-case main registry; owner boundaries and every returned numerical field match the direct MRST utility fixture |
| `meshHFsystem.m` | `utils/mesh_hf_system.py` | Implemented | Fracture-aligned tensor coordinates tested; direct topology fixture pending |
| `plotFracSystem.m` | `utils/plot_frac_system.py` | Implemented | Native matrix wireframe, selected fracture polygons/labels, well trajectories, intersection markers and global fracture-completion centroid lookup return MRST-shaped handle groups; renderer-safe structural tests pass, visual MATLAB fixture pending |
| `plotfracSystem_NF.m` | `utils/plot_frac_system_nf.py` | Implemented | Hydraulic/natural classification, separate colors, red wells, selection and shared interactive plotter are tested; visual MATLAB fixture pending |
| `processStochFracs.m` | `utils/process_stochastic_fractures.py` | Implemented | NaN/area/planarity filtering tested; MATLAB classification fixture pending |

### `examples`

| MRST owner | Required Python workflow | Status | Missing behavior/evidence |
|---|---|---|---|
| `compositional3D.m` | `examples/compositional_3d.py` | Implemented | Three exact normalized fracture planes, matrix/fracture and cross-fracture NNCs, global operators, water injector, BHP producer, verysimple mixture and five-year schedule are native. Reduced 53-cell nonlinear waterflood passes with correct control signs; full-size MATLAB time-history regression pending |
| `diffusion.m` | `examples/diffusion.py` | Implemented | Self-contained owner-level port explicitly constructs the Barnett grid, fractures, well mapping, EDFM operators, diffusion coefficients/tortuosity, enabled/disabled models, initial states and paired schedules; reduced nonlinear schedule and nonzero gradient diagnostics pass, full 15-year MATLAB regression pending |
| `eagleFord.m` | `examples/eagle_ford.py` | Implemented | Exact short-field matrix/HF/NF/well inputs, bundled 75+75 realization, bounded/full preprocessing, geometry-derived WI, well-fracture completions, projected operators, fluid/model/state and schedule are native; complete pEDFM production history and MATLAB certification remain |
| `eagleFordEDFM.m` | `examples/eagle_ford_edfm.py` | Implemented | Self-contained owner-level port explicitly constructs the short-field parameters, fractures, matrix grid, EDFM/NNC topology, well mapping, shale EDFM operators, fluid/model/state and 15-year schedule; a live reduced BHP-production step and the original first 79.1-second report step converge, while complete production-history and MATLAB certification remain |
| `eagleFordEDFM_MEX.m` | `examples/eagle_ford_edfm_accelerated.py` | Implemented | Self-contained owner-level accelerated EDFM port explicitly constructs geometry, fractures/NNCs, shale EDFM operators, well, model/state/schedule and installs the native compiled Numba row-major diagonal backend; kernel activation and setup are tested, but first-call/runtime benchmarking and nonlinear history certification remain |
| `eagleFordEDFMfull.m` | `examples/eagle_ford_edfm_full.py` | Implemented | Self-contained owner-level full-field EDFM port explicitly owns the seven-stage parameters, 21 HFs, bundled 4,200-NF mapping, 920 m well, EDFM/NNC preprocessing, shale operators, fluid/model/state and 15-year schedule; full preprocessing/simulation acceptance remains |
| `eagleFord_MEX.m` | `examples/eagle_ford_accelerated.py` | Implemented | Self-contained owner-level accelerated pEDFM port explicitly constructs geometry, fractures, NNCs, projected operators, well, model/state/schedule and installs the native compiled Numba row-major diagonal backend; kernel activation and setup are tested, but first-call/runtime benchmarking and nonlinear history certification remain |
| `eagleFordfull.m` | `examples/eagle_ford_full.py` | Implemented | Self-contained owner-level pEDFM port explicitly owns the full field parameters, 21 HFs, bundled 4,200-NF mapping, 920 m well, EDFM/NNC preprocessing, projected operators, fluid/model/state and 15-year schedule; full preprocessing/simulation acceptance remains |
| `gangi.m` | `examples/gangi.py` | Implemented | Self-contained owner-level port explicitly constructs the Barnett grid, fractures, well mapping, EDFM operators, MRST Gangi parameters/function, enabled/disabled models, states and paired schedules; reduced nonlinear schedule and matrix-only multiplier diagnostics pass, full 15-year MATLAB regression pending |
| `gangiPlot.m` | `examples/gangi_plot.py` | Implemented | Reusable workflow returns the exact 50-sample MRST pressure/correction curve and optionally renders/saves its annotated plot; numerical regression passes |
| `langmuirPlot.m` | `examples/langmuir_plot.py` | Implemented | Reusable workflow returns the exact 100-sample MRST pressure/sorbed-density curve and optionally renders/saves its annotated plot; numerical regression passes |
| `sorption.m` | `examples/sorption.py` | Implemented | Builds the fractured Barnett producer and paired shale/base models with exact isotherms and grain volume; reduced nonlinear schedule and matrix-only storage diagnostics pass, full 15-year MATLAB regression pending |
| `stochasticFracs.m` | `examples/stochastic_fractures.py` | Implemented | Exact bundled 350+350 polygon realization, typed set metadata, filtering/classification, seeded external-generator contract, optional hydraulic/natural rendering and bounded/full EDFM preprocessing are tested; ADFNE stereonet presentation remains optional external functionality |
| `Contents.m` | `examples/contents.py` | Catalog | Ordered machine-readable owner/module/description/section catalog matches the MRST reference contents |

#### Example data

| MRST asset | Status | Required action |
|---|---|---|
| `EagleFordEDFMGrid.mat` | Packaged | SHA-256 matches MRST source; typed grid conversion still pending |
| `EagleFordFracs.mat` | Packaged and typed | SHA-256 matches MRST source; all 150 polygons convert to natural fractures with alternating MRST petrophysics |
| `EagleFordGrid.mat` | Packaged | SHA-256 matches MRST source; typed grid conversion still pending |
| `EagleFordfullFracs.mat` | Packaged and typed | SHA-256 matches MRST source; all 4,200 polygons feed both full-field entrypoints |
| `StochasticFracs.mat` | Packaged and typed | SHA-256 matches MRST source; all 700 closing-row polygons convert to typed hydraulic/natural planes and optional EDFM input |

### `SPE201243PA`

| MRST owner | Required Python workflow | Status | Missing behavior/evidence |
|---|---|---|---|
| `Fig10to13TeneCaseLowKf.m` | `spe201243pa/fig_10_to_13_tene_low_kf.py` | Implemented | Exact 27x27x3 matrix, crossed sealing fractures, EDFM/pEDFM stacks, 225x225x2 resolved reference option, compressible-water boundaries and aligned runner are native; reduced three-method solve passes, full 60-day profiles/MATLAB fixture remain |
| `Fig10to13TeneCasesHighKf.m` | `spe201243pa/fig_10_to_13_tene_high_kf.py` | Implemented | Reuses the same native comparison workflow with exact 1,000-darcy fracture contrast; reduced three-method solve passes, full profiles/MATLAB fixture remain |
| `Fig15_17_18_validationCase2.m` | `spe201243pa/fig_15_17_18_validation_case_2.py` | Implemented | Exact graded 6,405-cell matrix, two barriers plus central 50-D fracture, pEDFM/EDFM/explicit models, central BHP completion and CMG timestep schedule are native; live three-method acceptance passes, full CMG history/MATLAB profiles remain |
| `Fig16_validationCase2_structured.m` | `spe201243pa/fig_16_validation_case_2_structured.py` | Implemented | Distinct 6,615-cell structured matrix input feeds the same three-method physical comparison; setup/operator/well mapping tests pass, full history/MATLAB profiles remain |
| `Fig21_22_23_bakken_edfm.m` | `spe201243pa/fig_21_22_23_bakken_edfm.py` | Implemented | Eight stages/24 HFs, 4,004-cell matrix, seeded 500+500 native DFN, horizontal producer, Bakken fluid and 30-year schedule are native; hydraulic-network solve passes. MRST does not distribute its ADFNE realization, so exact DFN/full history validation requires a supplied generator or fixture |
| `Fig21_22_23_bakken_pedfm.m` | `spe201243pa/fig_21_22_23_bakken_pedfm.py` | Implemented | Same typed case with the distinct projected operator graph and positive fracture PV; hydraulic-network first report step converges. Exact ADFNE/full production-history MATLAB certification remains |
| `MulistagePlanarNonPlanarHF.m` | `spe201243pa/multistage_planar_nonplanar_hf.py` | Validated | Scalar/per-stage spacing and dimensions, inclined vertices, stage metadata and centroid progression match a direct MRST fixture |
| `StructuredGridFractureonFace.m` | `spe201243pa/structured_grid_fracture_on_face.py` | Validated | Stage-aware fine/interstage/boundary x refinement and tensor coordinates place every fracture exactly on a face and match direct MRST |
| `explicitStencil.m` | `spe201243pa/explicit_stencil.py` | Validated | Geometric/log/linear/cartesian grading, tip refinement, zero-based fracture cells and center well are implemented; default geometric grid coordinates and IDs match direct MRST |
| `explicitStencil3Fracs.m` | `spe201243pa/explicit_stencil_3_fracs.py` | Validated | Three aperture-resolved fracture bands, outer/inter-fracture grading and middle well selection are implemented; geometric coordinates and IDs match direct MRST |
| `CMG_timestep2.csv` | Packaged | SHA-256 matches MRST source and loader preserves its original report sequence |

## Reusable PyMRST prerequisites

| Existing module | Potential contribution | Current shale implication |
|---|---|---|
| `pymrst.modules.hfm` | Consolidated fracture framework for shale: geometry, gridding, NNC and operator primitives | Required dependency; implemented subsets are certified for shale use while broader MRST `hfm_v2` behavior remains unported |
| `pymrst.modules.dfm` | Conforming-workflow façade | No projection-based shale parity |
| `pymrst.modules.compositional` | EOS, flash, natural/overall-variable models, AD | Required base; shale model extends its live natural-variable assembly |
| `pymrst.modules.ad_core` | AD variables, equations, nonlinear simulation | Required base; sorption residual and Jacobians now use its AD path |
| grid/plotting/well utilities | Geometry, visualization and common well structures | Must be certified against shale-specific use |

Reuse should occur through imports from `hfm` into final shale owners. Copying
generic kernels into `shale/__init__.py` or adding aliases does not close a
parity row.

## Validation parity

| Validation level | Current evidence | Status |
|---|---|---|
| Import | `validate_phase106.py` imports shale and checks one attribute | Surface only |
| Shale utility/NNC/well/fluid/equation/example tests | 116 focused tests in the reproducible headless qualification gate | Passing |
| Owner/data layout audit | 48/48 MRST owners present; all 47 executable owners referenced by tests; 6/6 assets byte-certified | Passing |
| 3D shale topology fixtures | Direct MRST operator-pipeline fixture plus Cartesian, oblique and general-polyhedral Python cases | Partial breadth |
| pEDFM projection fixtures | Direct MRST centered fixture plus exact overlap-union and affine-skew regressions pass | Validated |
| Well/WI fixtures | Direct MRST fracture WI, intersection and `addWellShale` structure fixture passes; matrix/mixed option breadth is additionally Python-tested | Validated |
| Shale residual/Jacobian fixtures | Sorption insertion, diffusion AD propagation, Gangi matrix masking and non-Darcy phase hooks tested in Python; direct assembled MATLAB fixture absent | Partial |
| Sorption/diffusion/Gangi tests | All mechanisms are connected to accumulation or flux assembly; paired fractured-model setup and reduced nonlinear schedules run for all three, with active-kernel diagnostics; complete MATLAB time-history fixtures remain | Implemented |
| Closed-system material balance | Componentwise diffusion divergence is certified exactly conservative; complete coupled inventory histories remain | Partial |
| Eagle Ford regression | None found | Missing |
| SPE-201243-PA regression | Direct geometry fixture covers all four deterministic grid/fracture owners; six simulation workflows remain | Partial |
| CMG comparison | Original CSV packaged; comparison runner absent | Partial |
| MATLAB cross-validation | Direct MRST fixtures cover nine deterministic owners, including every case in both fluid catalogs and all four SPE geometry builders | Partial |

## Highest-priority blockers

1. **Certified consolidated HFM foundation and native 3D shale grid contract:** native
   rectangular fracture preprocessing and the complete matrix-to-operator
   pipeline now exists, including exact convex polygon clipping against actual
   Cartesian/tetrahedral/polyhedral cell edges, heterogeneous properties,
   stable repeated calls, and physical fracture node/face/cell topology.
   Direct MATLAB topology certification remains.
2. **Projection-based NNC certification breadth:** the owner now has exact
   polygon union, affine-skew support and a direct centered MATLAB fixture;
   additional direct MATLAB overlap/skew fixtures remain desirable.
3. **Shale operators and well completion:** native EDFM/pEDFM operators now
   expose the full AD simulation contract, and shale WI/well-fracture mapping
   match MRST; the expanded `addWellShale` option surface is Python-tested and
   its fracture-completion structure is directly certified against MRST.
4. **Shale compositional certification:** all four mechanisms now enter the
   natural-variable accumulation/flux assembly. Direct MATLAB residual,
   Jacobian, conservation and coupled-timestep fixtures remain.
5. **Reference workflow certification:** all executable Eagle Ford and SPE/CMG
   owners are native and their reduced live paths are covered. Complete report
   histories, exact pressure/rate profiles and full-field runtime evidence
   remain; Bakken also needs an archived ADFNE realization for exact geometry.
6. **Direct certification:** thirteen owners have direct MATLAB fixtures.
   Remaining equation residual/Jacobian, sparse operator and complete
   end-to-end history comparisons are the principal parity gap.

## Completion rule

The executable owner inventory is now fully present, but that alone is not the
completion claim. Final parity requires focused tests, MATLAB numerical
comparison, packaged/provenance-controlled data, and successful full-history
end-to-end examples as specified in
`shale_implementation.md`.
