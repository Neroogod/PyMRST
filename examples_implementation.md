# Porting `mrst-2024a-Geomech/examples`

A plan for porting MRST's top-level `examples/` tree end to end, at MRST-shaped
paths, with the dependency work stated up front because it dominates the effort.

The conclusion first: the visible work is 17 files and about 2920 lines, but the
eight missing dependencies behind them come to 1885 lines, and one of them lives
in a module PyMRST does not have. Counting only the examples understates the job
by roughly two thirds. The tree splits cleanly into four phases, and the first
two can be finished without touching any absent module.

Progress: Phases 1 and 2 are done apart from `gridTutorialCornerPoint`, giving
8 of 17 files and 55 self-tests. See the status notes under each phase.

Status and inventory live in `examples_parity.md`. This document covers what to
build and in what order.

## Why the raw file count misleads

`examples/` holds 20 `.m` files. Three are `Contents.m` catalogs, leaving 17
executable tutorials. That looks like a small job next to `pymrst/utils`, which
ran to 156 files.

It is not, for three reasons.

**The dependencies outweigh the examples.** Cross-checking every call in the tree
against MRST's function inventory and against PyMRST's modules and definitions
leaves eight functions with no counterpart under any spelling: `processWells`
(804 lines), `initEclipseRock` (336), `initEclipseGrid` (236), `getSPE10rock`
(134), `processPartition` (117), `explosionView` (102), `partitionLayers` (100),
and `compressRock` (56).

**One dependency has no home.** `coarsegrid` is absent from PyMRST, and it owns
`processPartition`, `partitionLayers`, and `explosionView`. `showSAIGUP` needs
all three.

**Two datasets are not present.** Johansen, Egg, SPE10, and Norne are downloaded
locally, at 233, 166, 106, and 18 MB. SAIGUP and CaseB4 are not. Descriptors for
both exist under `pymrst/utils/dataset_manager/datasets/`, so the download path
exists and only needs exercising.

Against that, the tutorials themselves are mostly thin. They call library
functions and plot the results, so once a dependency is present the example above
it is short work.

## A note on measuring the gap

An automated scan of this tree will over-report missing dependencies unless its
name matching is acronym-aware. Naive camel-to-snake conversion turns
`processGRDECL` into `process_g_r_d_e_c_l`, which matches nothing, and the same
happens to `readGRDECL`, `initDeckADIFluid`, and `initVariablesADI`. All four are
present in PyMRST. Any audit script written for this tree should special-case
runs of capitals, and its first output should be checked by hand against a few
known-present functions.

## Phase 1: root tutorials

Five files, 1015 lines.

| MRST | PyMRST target | Tests |
|---|---|---|
| `flowSolverTutorial1.m` | `pymrst/examples/flow_solver_tutorial1.py` | 5 |
| `tutorialAD.m` | `pymrst/examples/tutorial_ad.py` | 6 |
| `tutorialBasicObjects.m` | `pymrst/examples/tutorial_basic_objects.py` | 7 |
| `flowSolverTutorialAD.m` | `pymrst/examples/flow_solver_tutorial_ad.py` | 6 |
| `tutorialPlotting.m` | `pymrst/examples/tutorial_plotting.py` | 7 |

### Phase 1 status: complete, and the "no blockers" claim above was wrong

All five are ported, carry 31 self-tests between them, and are picked up by
`tests/test_all.py::test_example_drivers`.

**The plan said this phase had no blocking dependencies. It did.**
`tutorialBasicObjects` calls `compressRock` twice, once on a rock restricted to
a reduced grid and once on a rock read from a deck. The static analysis behind
the estimate found `compressRock` and attributed it to the Egg examples, having
identified the function without identifying every caller. `compressRock` was
therefore ported early, at 56 lines the smallest of the eight, to
`pymrst/modules/deckformat/params/rock/compress_rock.py` with six self-tests.
Phase 3's dependency total drops from 1432 lines to 1376.

The lesson generalises to the phases below: a dependency list built from call
extraction tells you which functions are missing, not which files need them.
Expect at least one more such surprise in Phases 2 to 4.

What each tutorial is checked against, since assertions matter more than the
figures MRST ends on:

* **`flow_solver_tutorial1`** compares against the closed form
  `p = p_top + rho*g*z` at machine precision, and requires every flux to be
  zero rather than small. Density and column height are varied to confirm both
  scale the answer as they should.
* **`tutorial_ad`** checks AD against the hand-worked derivative of
  `3*exp(-x*y)` to 1e-14, confirms the AD Jacobian of a linear system is
  literally the coefficient matrix, and requires the AD and hand-coded Newton
  solves of the Rosenbrock gradient to take the *same number of iterations*,
  which is evidence the Jacobians agree rather than merely both converging.
  MRST's timing table is kept as a reported measurement rather than an
  assertion, since timings are machine-dependent: vectorised AD came out 148
  times faster than the element-by-element loop.
* **`tutorial_basic_objects`** verifies that removing a 3x3x6 block removes
  exactly 54 cells, that the compressed rock lines up row for row with the
  reduced grid, that Kozeny-Carman permeability is monotone in porosity, and
  that pore volume equals `volume * poro * ntg`.
* **`flow_solver_tutorial_ad`** closes a mass balance: the change in
  `sum(pv(p)*rho(p))` against the accumulated surface production matches to
  3.6e-13 relative. It also checks the ODE initial condition satisfies
  `dp/dz = g*rho(p)`, and that driving fluid compressibility to zero recovers
  the linear profile.
* **`tutorial_plotting`** keeps every computation and makes the figures
  optional, checking that pressure stays inside the two bhp controls, that
  saturations close to one and stay in `[0, 1]`, and that the front advances
  away from the injector.

Two implementation notes worth carrying forward:

**Index conventions bite here.** MRST marks an exterior face with neighbour 0
and writes `all(N ~= 0, 2)`; PyMRST uses -1. Testing the wrong sentinel keeps
the boundary faces in the interior set and lets flow leave the domain, which
still produces a plausible picture. `flow_solver_tutorial_ad` asserts the
exterior faces are gone rather than assuming.

**`combine_equations`, not `vertcat_adi`.** MRST's `cat` both stacks residuals
and assembles the per-variable Jacobian blocks into one square matrix.
`vertcat_adi` only stacks values, leaving separate blocks and a non-square
system. `combine_equations` is the true counterpart.

**Interactive prompts must be suppressed.** `get_dataset_path` prompts on the
terminal and waits, which hangs any automated run. `tutorial_basic_objects`
passes `download=False` so the BedModel2 section reports absence instead of
blocking. Any later example touching a dataset needs the same care.

## Phase 2: `grids/`

Four files, 680 lines. Three have no blockers.

| MRST | PyMRST target | Blocked by |
|---|---|---|
| `grids/gridTutorialIntro.m` | `pymrst/examples/grids/grid_tutorial_intro.py` | none |
| `grids/gridTutorialStruct.m` | `pymrst/examples/grids/grid_tutorial_struct.py` | none |
| `grids/gridTutorialUnstruct.m` | `pymrst/examples/grids/grid_tutorial_unstruct.py` | none |
| `grids/gridTutorialCornerPoint.m` | `pymrst/examples/grids/grid_tutorial_corner_point.py` | `initEclipseGrid`, `initEclipseRock` |

`gridTutorialStruct` calls ten MRST functions and PyMRST has every one, so it is
the cheapest file in the tree and a good second target after Phase 1.

### Phase 2 status: three of four complete

`grid_tutorial_struct`, `grid_tutorial_unstruct`, and `grid_tutorial_intro` are
ported with 8, 7, and 9 self-tests. `gridTutorialCornerPoint` remains, waiting on
`initEclipseGrid` and `initEclipseRock` from Phase 3, as planned.

The prediction that another dependency surprise would appear did not come true.
No new blockers were found; every routine these three tutorials need was already
present. What appeared instead were three **convention** mismatches, none of
which a missing-function scan can detect, and each of which produces a wrong
answer rather than an error:

**`mcolon` is half-open here and inclusive in MRST.** MRST writes
`mcolon(nodePos(fI), nodePos(fI) + 1)` to collect both endpoints of a 2-D face.
PyMRST's `mcolon` is `[lo, hi)`, so the literal translation marks only the first
node of each boundary face and leaves the other half free to move. The
perturbation test caught it: the grid's bounding box changed, meaning the domain
outline had moved rather than the mesh inside it. `nodePos[bf + 1]` is both
correct and dimension-independent, where MRST's `+1` assumes two nodes per face
and so is 2-D only.

**`hexahedral_grid` reverses MRST's argument order.** PyMRST takes
`(hexes, coords)`; MRST's `hexahedralGrid` takes `(P, H)`, points first. The
PyMRST docstring says so, and passing them in MRST's order fails immediately
with an out-of-range node index, which is the good case.

**The boundary sentinel differs, again.** MRST finds the boundary with
`any(neighbors == 0, 2)`; PyMRST uses -1, because 0 is a valid 0-based cell
index. `grid_tutorial_intro` asserts that the two tests disagree on its grid, so
the difference is documented by a failing comparison rather than by a comment.

One deviation is deliberate and recorded in the file: MRST's unstructured
tutorial loads `seamount`, sample data shipped with MATLAB rather than with MRST.
There is nothing to port, so `scattered_points` generates a comparable irregular
cloud. The construction being shown, a Delaunay triangulation of scattered
data, is unchanged, and the tests check structural properties rather than
coordinates.

Two assertions worth noting, because both check what a picture only suggests.
`grid_tutorial_struct` verifies that perturbing interior nodes leaves the
bounding box exactly unchanged, which is the difference between deforming a mesh
and deforming its domain. `grid_tutorial_unstruct` verifies that radial grading
actually refines: cells within 0.4 of a well centre came out 194 times smaller
than background cells 3 to 6 units away.

Grid tutorials validate well against structure checksums rather than figures.
Cell and face counts, `facePos` monotonicity, volumes, centroids, areas, and node
coordinates all compare exactly, and the MATLAB file bridge already showed this
works: those quantities matched MATLAB to a relative 5.6e-14, and the `neighbors`
array summed to precisely `matlab_sum - 2 * n_faces` after index conversion.

## Phase 3: deck dependencies, then the Egg examples

Three functions remaining, 1376 lines, unlocking three examples worth 350 lines. `compressRock` was pulled into Phase 1; see the status note there.

Port in this order, since each depends on the one above:

1. ~~`compressRock` (56 lines)~~ **done in Phase 1**, at
   `pymrst/modules/deckformat/params/rock/compress_rock.py`
2. `initEclipseRock` (336 lines, `deckformat/params/rock`)
3. `initEclipseGrid` (236 lines, `deckformat/grid`)
4. `processWells` (804 lines, `deckformat/params/wells_and_bc`)

All four belong in `pymrst/modules/deckformat/`, which exists, so they go to
MRST-shaped paths inside a module already present. `processWells` is the single
largest item in the whole plan and deserves its own validator rather than being
checked only through the examples that call it.

Completing this phase also unblocks `gridTutorialCornerPoint` from Phase 2, so
that file should be finished here rather than left behind.

The Egg examples then follow:

| MRST | PyMRST target |
|---|---|
| `data/Egg/.../Reading_deck_Egg_model_Fully_Implicit.m` | `pymrst/examples/data/egg/reading_deck_egg_model_fully_implicit.py` |
| `data/Egg/.../Reading_deck_Egg_model_Sequential.m` | `pymrst/examples/data/egg/reading_deck_egg_model_sequential.py` |
| `data/Egg/.../StoreProduction.m` | `pymrst/examples/data/egg/store_production.py` |

The Egg dataset is already downloaded at 166 MB, and PyMRST has
`modules/deckformat/egg/read_egg_deck.py`, so the reading path is partly built.
Both Egg runs produce well production histories, which are proper numerical
validation targets.

Note the layout question this phase raises. MRST keeps these three files under
`examples/data/Egg/Egg_Model_Data_Files_v2/MRST/`, which is a dataset payload
directory rather than a code directory. Mirroring it exactly puts Python source
inside a data tree. The alternative is `pymrst/examples/egg/`, which is cleaner
but breaks the layout contract. Recommendation: mirror MRST, since the contract
has been held everywhere else and `ported-elsewhere` deviations were driven to
zero in `pymrst/utils`. Record the decision either way.

## Phase 4: `datasets/`

Five files, 875 lines, and the phase with the most external dependencies.

| MRST | Blocked by | Data |
|---|---|---|
| `datasets/showNorne.m` | none | present, 18 MB |
| `datasets/showJohansen.m` | none | present, 233 MB |
| `datasets/showSPE10.m` | `getSPE10rock` | present, 106 MB |
| `datasets/showCaseB4.m` | none | download required |
| `datasets/showSAIGUP.m` | `coarsegrid` module | download required |

Order: Norne and Johansen first, since both are unblocked and their data is on
disk. `showSPE10` follows once `getSPE10rock` (134 lines) is ported.
`showCaseB4` needs only a download.

`getSPE10rock` raises a placement question. MRST keeps it in a top-level
`modules/spe10`, which PyMRST does not have: its SPE10 code sits under
`pymrst/modules/deckformat/spe10` instead, with a second copy of example material
under `ad_blackoil/examples/spe10`. Putting `getSPE10rock` at the MRST-shaped
path means creating `pymrst/modules/spe10/`, which then coexists with the
`deckformat/spe10` code already there. Either consolidate the two or record this
as a deliberate `ported-elsewhere` entry. The same question was driven to zero
deviations in `pymrst/utils`, so consolidation matches precedent.

`showSAIGUP` is the hardest file in the tree and should be taken last. It needs
`processPartition`, `partitionLayers`, and `explosionView`, all from `coarsegrid`,
which does not exist in PyMRST. Porting three functions to create a module is
reasonable; porting `coarsegrid` properly is a separate project. Two defensible
choices exist:

* port the three functions into a minimal `pymrst/modules/coarsegrid/` and accept
  that the module is a stub serving one example, or
* exclude `showSAIGUP` with a stated reason until `coarsegrid` is ported for its
  own sake.

The second is more honest if `coarsegrid` is not otherwise wanted, since a
three-function module invites the assumption that the rest is present. This is a
decision to record rather than to make silently.

## Effort summary

| Phase | Files | Example lines | Dependency lines | Blockers |
|---|---|---|---|---|
| 1, root tutorials | 5 | 1015 | 56 | **complete** |
| 2, grids | 4 | 680 | 0 | **3 of 4 complete**; corner-point waits on Phase 3 |
| 3, deck plus Egg | 3 | 350 | 1376 | none once ported |
| 4, datasets | 5 | 875 | 453 | `coarsegrid` absent, 2 downloads |
| **total** | **17** | **2920** | **1885** | **8 of 17 ported** |

Phases 1 and 2 together deliver 8 of 17 files with no new library code, which is
the reason to run them first: they establish the package, the test wiring, and
the validation pattern before any dependency work begins.

## What "done" requires

Beyond the files themselves:

1. A `pymrst/examples/` package mirroring MRST's layout, wired into the standard
   test sweep the way module examples already are, through
   `tests/test_all.py::test_example_drivers`.
2. `__main__` self-tests in every ported file, matching the house convention that
   now reaches 45% of the package.
3. MATLAB validators for the numerically meaningful examples: both flow-solver
   tutorials, `tutorialAD`, and both Egg runs.
4. Grid-structure checksum validation for the four grid tutorials.
5. The two absent datasets fetched through the existing dataset manager, so the
   examples are runnable from a clean checkout.
6. A decision recorded for the two layout questions above: whether Egg examples
   mirror MRST's data-directory placement, and whether `showSAIGUP` gets a stub
   `coarsegrid` or a stated exclusion.

## A caveat on this plan

The dependency analysis behind these numbers is static. Calls were extracted from
each example and matched against MRST's function inventory and PyMRST's modules
and definitions, which finds missing functions but not behavioural gaps: a
function present in PyMRST may still lack an option one of these tutorials uses.
The 1885-line figure is therefore a floor on the dependency work, not a
prediction. Expect Phase 3 in particular to grow, since `processWells` at 804
lines is large enough to hide option coverage that only running the Egg examples
will reveal.
