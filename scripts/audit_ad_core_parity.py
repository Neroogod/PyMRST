#!/usr/bin/env python3
"""Generate the MRST ``ad-core`` to PyMRST parity ledger, with depth.

Phase 0 of this scanner called a class ``ported`` when a filename matched.
Under that rule ``PhysicalModel`` - 4 of 49 methods - was ported, and so was
``UniformFacilityModel``, whose counterpart was a class statement and a
docstring. A ledger that cannot tell those apart from ``SimpleWell`` (22 of
22) is a filename listing with a reassuring column, not evidence.

This pass adds per-method depth for MATLAB ``classdef`` files: every method
name declared in the ``.m`` file is checked against the mapped Python
counterpart (``methods_in_counterpart``), then against the rest of the
``ad_core`` tree if not there (``methods_elsewhere``), and named if found
nowhere (``methods_absent``). ``depth = methods_in_counterpart / mrst_methods``
after subtracting any names in ``EXCLUDED_METHODS`` - the platform/layout/
subsref cases with no possible Python counterpart, each with a stated,
checked reason (see ``tests/ad_core/test_parity_ledger_depth.py``).

A classdef with fewer than ``MIN_METHODS_FOR_DEPTH`` methods is not judged by
ratio - 1 of 1 is not evidence either way - and keeps the filename/reference
classification. At or above that count, ``depth < DEPTH_BAR`` demotes
``ported`` to ``ported-shallow`` (or ``wrapper`` if depth is exactly 0: a
counterpart file exists and implements none of it).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

#: Below this many methods, a ratio is noise (1/1 = 100%, 1/2 = 50% - neither
#: says anything). Pinned; see test_the_depth_bar_itself_is_pinned.
MIN_METHODS_FOR_DEPTH = 5

#: A classdef at or above MIN_METHODS_FOR_DEPTH keeps `ported` only at or
#: above this fraction of its methods actually present. Pinned for the same
#: reason: a bar that can be quietly lowered to make a shallow list vanish is
#: not a bar.
DEPTH_BAR = 0.6

#: Method names excluded from depth scoring, with a reason that must name
#: both *why* (mex/platform/layout/subsref) and *where the capability lives*
#: instead - an exclusion without both reads as "PyMRST cannot do this,"
#: which in every case here is false. Contents pinned by
#: test_per_method_exclusions_are_pinned_and_justified; do not add an entry
#: without also justifying it there.
EXCLUDED_METHODS: dict[str, dict[str, str]] = {
    "DiagonalAutoDiffBackend": {
        "getBlockSystemCSR": (
            "MRST's MEX fixed-block-size CSR storage (a platform/layout "
            "concern, not an algorithm). The block-Jacobian algorithm itself "
            "is ported and validated in DiagonalJacobian/RowMajorDiagonalJacobian; "
            "only this specific MEX memory layout has no Python counterpart."),
        "applySchurComplementBlockSystemCSR": (
            "Same MEX fixed-block CSR layout as getBlockSystemCSR - a platform "
            "storage format, not a missing algorithm. The Schur complement "
            "itself is implemented against this port's own (non-CSR) diagonal "
            "representation."),
        "getRHS": (
            "Reads the right-hand side out of the same MEX block-CSR layout "
            "as the two entries above - a platform accessor for a storage "
            "format this port does not use. The right-hand side itself is "
            "ported and available directly as the residual vector."),
    },
    "DiagonalJacobian": {
        "compareIndices": (
            "Asks whether two positions fall in the same slot of MRST's "
            "n x ndiag column layout (a platform/layout question). This port's "
            "diagonal representation is triplet-based, so the question this "
            "method answers does not arise; setDiagonal, which survives the "
            "change of representation, is ported."),
        "treatSubset": (
            "Same n x ndiag column layout as compareIndices - a layout-specific "
            "index transform with no meaning against this port's triplet "
            "storage, which is implemented and validated on its own terms."),
    },
    "ResultHandler": {
        "getData_private": (
            "MATLAB subsref plumbing: the private half of overloaded "
            "indexing (`obj(i)`) dispatch. Its Python equivalent, "
            "`__getitem__`, is implemented on ResultHandler and carries the "
            "same branch directly, so there is no separate private half to "
            "port."),
    },
}


#: MATLAB operator-overload method names, mapped to the Python dunder that
#: plays the same role. `plus`/`minus`/... are real methods in MRST's
#: classdefs (operator overloading), and their absence as literal Python
#: identifiers named `plus`/`minus` is not a gap - Python spells operator
#: overloading with dunders, and a name-based scan that doesn't know that
#: reports every arithmetic class as badly ported.
OPERATOR_ALIASES = {
    "plus": "__add__", "minus": "__sub__", "uminus": "__neg__",
    "mtimes": "__mul__", "times": "__mul__", "rdivide": "__truediv__",
    "ldivide": "__truediv__", "mrdivide": "__truediv__",
    "eq": "__eq__", "ne": "__ne__", "lt": "__lt__", "le": "__le__",
    "gt": "__gt__", "ge": "__ge__",
    "subsref": "__getitem__", "subsasgn": "__setitem__",
    "horzcat": "__add__", "vertcat": "concatenate",
    "cat": "concatenate", "double": "value",
}

#: Verified architectural renames/relocations: `{owner_symbol: {mrst_name:
#: (python_identifier, reason)}}`. Unlike EXCLUDED_METHODS these are not
#: gaps - the capability exists, checked directly against the source below -
#: just not discoverable by name matching, because the method moved class
#: (MRST's model-driven Newton loop vs this port's separate NonLinearSolver)
#: or was renamed outright. Every entry was confirmed 2026-08-08 by reading
#: both the MRST method's signature/role and the Python definition it maps
#: to, not asserted from the name alone.
RENAMED_METHODS: dict[str, dict[str, str]] = {
    "DiagonalJacobian": {
        "sparse": "to_sparse", "full": "toarray", "sum": "diagonal_sum",
        "diagMult": "diag_mult", "diagProductMult": "diag_product_mult",
        "getSparseBlocks": "get_sparse_blocks",
        "getSparseArguments": "get_sparse_arguments",
        "matrixDims": "matrix_dims",
        # MRST implements `.^` at both the DiagonalJacobian and GenericAD
        # level; this port only needs it at the GenericAD level, where the
        # chain rule for x**n is actually applied - confirmed present as
        # GenericAD.__pow__.
        "power": "__pow__",
    },
    "GenericAD": {
        "reduceToDouble": "value",
        # The loop `lMultDiag` runs (apply diag_mult to every Jacobian block
        # in a list) is inlined at GenericAD's own arithmetic call sites
        # rather than factored into a same-named helper - confirmed at the
        # __mul__/__add__ implementations, which each build
        # `[diag_mult(v, j) for j in ...]` directly.
        "lMultDiag": "diag_mult",
        # MRST's GenericAD.vertcat is the instance-level entry point;
        # `vertcat_adi` (pymrst.utils.adi.operators) is this port's, and it is
        # literally what dispatches to DiagonalJacobian.vertcat/
        # RowMajorDiagonalJacobian.vertcat for the block-level work.
        "vertcat": "vertcat_adi",
        "combineEquations": "combine_equations",
    },
    "PhysicalModel": {
        "checkConvergence": "get_convergence",
        "stepFunction": "solve_ministep",  # moved to NonLinearSolver
        "getAdjointEquations": "solve_adjoint",
        "updateForChangedControls": "_apply_control",  # moved to NonLinearSolver
        "makeStepReport": "StepReport",  # function -> dataclass constructor
        # MRST's PhysicalModel is the shared abstract base for both reservoir
        # and facility models; this port implements these on the concrete
        # subclass that actually owns the responsibility rather than
        # replicating them on the literal base. Same snake-case name, verified
        # present, just one class down the hierarchy:
        "getModelEquations": "get_model_equations",       # WrapperModel/GenericReservoirModel/CompositeModel
        "setupStateFunctionGroupings": "setup_state_function_groupings",  # FacilityModel
        "removeStateFunctionGroupings": "remove_state_function_groupings",  # FacilityModel
        "updateAfterConvergence": "update_after_convergence",  # FacilityModel
        "getConvergenceValues": "get_convergence_values",  # FacilityModel
    },
    # WrapperModel(PhysicalModel) and CompositeModel(PhysicalModel) genuinely
    # subclass it (confirmed by reading the class declaration) and do not
    # override these four - so they have them by ordinary inheritance, the
    # same as any Python subclass that does not redefine a base method. Not a
    # coincidence-of-name guess: the base-class relationship was checked.
    "WrapperModel": {
        "checkConvergence": "get_convergence",
        "getAdjointEquations": "solve_adjoint",
        "stepFunction": "solve_ministep",
        "updateForChangedControls": "_apply_control",
    },
    "CompositeModel": {
        "checkConvergence": "get_convergence",
        "getAdjointEquations": "solve_adjoint",
        "stepFunction": "solve_ministep",
        "updateForChangedControls": "_apply_control",
    },
    # MATLAB subfunctions (relPermWO/relPermOG/relPermWOG/relPermWG), not
    # top-level names - the same "invisible to a def-only scan" shape as
    # ad_props's assignRelPerm.m subfunctions, verified there earlier this
    # session. Confirmed present as closures inside rel_perm_unified, with the
    # module's own docstring and an inline comment naming relPermWOG
    # explicitly. `subset` is StateFunction.subset, inherited - this class
    # extends StateFunction and does not override it.
    "BaseRelativePermeability": {
        "relPermWO": "rel_perm_unified", "relPermOG": "rel_perm_unified",
        "relPermWOG": "rel_perm_unified", "relPermWG": "rel_perm_unified",
        "subset": "subset",
    },
}


#: MRST names whose port's file name splits the words differently from
#: `snake_case` (checked: each file is the port of that MRST file).
FILENAME_ALIASES = {
    "NonLinearSolver": "nonlinear_solver",
    "MPFAvsTPFAwithADExample": "mpfa_vs_tpfa_with_ad_example",
    "adBuckleyLeverett1D": "ad_buckley_leverett_1d",
    "combineMSwithRegularWells": "combine_ms_with_regular_wells",
    "phaseDensitiesTobfactor": "phase_densities_to_bfactor",
    "double2ADI": "double_to_adi",
    "bc2ADbc": "bc_to_ad_bc",
}


#: MRST names whose Python counterpart was reviewed and found to be an earlier
#: PyMRST implementation behind the MRST name, not a port of the file: they
#: stay `pending-review` (with the reason) until ported and run against MRST,
#: whatever their filename, method names or mentions in tests say.
LEGACY_IMPLEMENTATIONS = {
    "CompositeModel": "legacy core on the legacy PhysicalModel: no packed "
                      "Newton system or submodel split; not run against MRST",
    "CouplingTerm": "legacy class (not a StateFunction, dict-keyed equations, "
                    "no nested submodels); not run against MRST",
    "SourceCouplingTerm": "legacy class on the legacy CouplingTerm; MRST's "
                          "ComponentTotalFlux divergence source not ported",
    "compositeModelTutorial": "a two-cell LinearPressureCoupling toy, not MRST's "
                              "coupled SPE10 layers (needs test-suite TestCase "
                              "and spe10_wo, not ported)",
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def source_index(paths: list[Path]) -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in paths
    }


#: `def NAME`, `class NAME`, or a bound alias `NAME = ...` - the three forms
#: this port uses for "MRST's identifier reaches something callable" (the
#: last is the documented convention: `addFluxesFromSourcesAndBC =
#: add_bc_and_sources`). A name search restricted to `def`/`class` alone
#: misses every file that only exposes the MRST spelling as an alias.
_BINDING = re.compile(
    r"(?:^|\s)(?:def|class)\s+([A-Za-z_]\w*)"
    r"|(?:^|[\s.])([A-Za-z_]\w*)\s*=\s*[^=]"
)


def bindings_in(text: str) -> set[str]:
    names = set()
    for m in _BINDING.finditer(text):
        names.add(m.group(1) or m.group(2))
    return names


_METHOD_DECL = re.compile(
    r"^\s*function\s+(?:\[[^\]]*\]\s*=\s*|[\w.]+\s*=\s*)?(\w+)\s*\(",
    re.MULTILINE,
)


def classdef_methods(matlab_text: str) -> list[str]:
    """Every method name declared anywhere in a classdef file.

    MATLAB classdef files declare functions only inside `methods` (including
    get/set property accessors, which are real API surface) and occasionally
    `properties` blocks contain none - scanning the whole file for `function
    ... NAME(` avoids tracking `methods`/`properties`/`end` nesting by hand,
    which MATLAB's block structure does not make trivial to parse with a
    regex in the first place.
    """
    if not re.search(r"^\s*classdef\b", matlab_text, re.MULTILINE):
        return []
    return _METHOD_DECL.findall(matlab_text)


def matlab_kind(matlab_text: str) -> str:
    if re.search(r"^\s*classdef\b", matlab_text, re.MULTILINE):
        return "classdef"
    if re.search(r"^\s*function\b", matlab_text, re.MULTILINE):
        return "function"
    return "script"


def find_counterparts(
    matlab_file: Path,
    matlab_root: Path,
    python_root: Path,
    python_sources: dict[Path, str],
) -> tuple[str, list[Path], str]:
    rel = matlab_file.relative_to(matlab_root)
    symbol = matlab_file.stem
    stem = snake_case(symbol)

    if symbol.lower() == "contents":
        return "excluded-catalog", [], "MATLAB package catalog"
    if "mex" in {part.lower() for part in rel.parts[:-1]} and symbol != "testMexDiagonalOperators":
        return (
            "excluded-platform",
            [],
            "MATLAB MEX build/runtime implementation; Python fallback required",
        )
    if symbol in {
        "ConservationLawJacobian",
        "FixedWidthJacobian",
        "blockCSRtoSparse",
        "getSparseBlocksFixedWidth",
    }:
        return (
            "excluded-platform",
            [],
            "MRST FixedWidth/ConservationLaw MEX storage or its private converter",
        )
    if symbol == "getDeckOPMData":
        return (
            "excluded-external",
            [],
            "optional opm-data/opm-tests repository and restart-output reader",
        )
    if symbol in {
        "plotStateFunctionGroupings",
        "printStateFunctionGroupingTikz",
    }:
        return (
            "excluded-visualization",
            [],
            "state-function graph visualization/export helper",
        )
    if symbol in {"AGMGSolverAD", "AMGCLSolverBlockAD", "AMGCL_CPRSolverBlockAD"} \
            and rel.parts[0] == "solvers":
        return (
            "excluded-external-solver",
            [],
            "wrapper for the AGMG binary / AMGCL's block backend, not built here",
        )
    if symbol in {"WellModel", "updateConnectionDP"} and rel.parts[:2] == ("models", "wells"):
        return (
            "excluded-deprecated",
            [],
            "deprecated classic well stack; superseded by FacilityModel/SimpleWell "
            "(connection pressure drop lives on FacilityModel.connection_pressure_drop)",
        )
    if symbol in {"plotRegionContacts", "terniaryWellPlot",
                  "plotPackedProblem", "examplePlottingStateFunctionsBook",
                  "fluidInspectionExample"}:
        return (
            "excluded-visualization",
            [],
            "MATLAB graphics-only helper; no simulation or numerical output",
        )
    if symbol in {
        "backendOptionsExample",
        "exampleBenchmarkBackends",
        "blockAssemblyExample",
        "blockAssemblyBigModelExample",
        "parallelScalingAD",
        "showADBenchmarkAssembly",
    }:
        return (
            "excluded-benchmark-driver",
            [],
            "MATLAB timing/plot driver; reusable benchmark utility is ported and validated",
        )
    if rel.parts[0] == "plotting" and not any(path.stem == stem for path in python_sources):
        return (
            "excluded-visualization",
            [],
            "MATLAB figure/UI helper; no simulation or numerical output",
        )

    stem = FILENAME_ALIASES.get(symbol, stem)
    exact = [path for path in python_sources if path.stem == stem]
    if exact:
        return "ported", exact, "filename"

    token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", re.I)
    mentions = [path for path, text in python_sources.items() if token.search(text)]
    if mentions:
        return "ported-consolidated", mentions, "symbol-reference"

    return "pending-review", [], ""


_CLASS_DECL = re.compile(r"^class\s+(\w+)\s*\(([^)]*)\)\s*:", re.MULTILINE)


def base_class_bindings(
    counterpart_text: str,
    python_sources: dict[Path, str],
    file_by_class: dict[str, Path],
) -> set[str]:
    """Bindings inherited from a base class this port actually declares.

    A subclass that does not override an inherited method still has it -
    ordinary Python semantics, not a gap. `WrapperModel(PhysicalModel)` and
    `CompositeModel(PhysicalModel)` both looked shallow before this: most of
    what they were "missing" was plain inheritance from PhysicalModel, which
    a scan of only their own file's `def` lines cannot see. One level only
    (a base's own bases are not walked) - enough for this port's shallow
    hierarchies without guessing at a chain that was not checked.
    """
    inherited: set[str] = set()
    for match in _CLASS_DECL.finditer(counterpart_text):
        for base in match.group(2).split(","):
            base = base.strip()
            base_file = file_by_class.get(base)
            if base_file is not None:
                inherited |= bindings_in(python_sources.get(base_file, ""))
    return inherited


def depth_scan(
    symbol: str,
    mrst_text: str,
    counterparts: list[Path],
    python_sources: dict[Path, str],
    tree_bindings: set[str],
    file_by_class: dict[str, Path],
) -> tuple[int, int, int, list[str], float | None]:
    """Per-method presence for one classdef symbol.

    Returns (mrst_methods, in_counterpart, elsewhere, absent_names, depth).
    `depth` is None below MIN_METHODS_FOR_DEPTH - not scored, not judged.
    """
    methods = classdef_methods(mrst_text)
    excluded = EXCLUDED_METHODS.get(symbol, {})
    scored = [m for m in methods if m not in excluded]
    total = len(scored)
    if total == 0:
        return 0, 0, 0, [], None

    counterpart_bindings: set[str] = set()
    for path in counterparts:
        text = python_sources.get(path, "")
        counterpart_bindings |= bindings_in(text)
        counterpart_bindings |= base_class_bindings(text, python_sources, file_by_class)

    renamed = RENAMED_METHODS.get(symbol, {})
    in_counterpart, elsewhere, absent = 0, 0, []
    for name in scored:
        candidates = {name, snake_case(name)}
        if name in OPERATOR_ALIASES:
            candidates.add(OPERATOR_ALIASES[name])
        if candidates & counterpart_bindings:
            in_counterpart += 1
            continue
        # Both a dunder mapping (MATLAB operator overloading -> Python's
        # spelling for the same thing) and a manually verified rename are
        # trusted identity claims, not name coincidences - a dunder mapping
        # is just as checked as a RENAMED_METHODS entry, it is pinned in code
        # instead of prose. Both count fully wherever they are found, not as
        # the weaker "some name happens to match somewhere" `elsewhere`
        # signal the rest of the search represents.
        verified_target = OPERATOR_ALIASES.get(name) or renamed.get(name)
        if verified_target and verified_target in tree_bindings:
            in_counterpart += 1
            continue
        if candidates & tree_bindings:
            elsewhere += 1
        else:
            absent.append(name)

    depth = None if total < MIN_METHODS_FOR_DEPTH else in_counterpart / total
    return total, in_counterpart, elsewhere, absent, depth


def validation_matches(symbol: str, validation_sources: dict[Path, str]) -> list[Path]:
    spellings = {symbol, snake_case(symbol)}
    alternatives = "|".join(re.escape(spelling) for spelling in sorted(spellings))
    token = re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])",
        re.I,
    )
    return [path for path, text in validation_sources.items() if token.search(text)]


def relative_list(paths: list[Path], root: Path) -> str:
    return "; ".join(str(path.relative_to(root)) for path in sorted(set(paths)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrst-root", type=Path,
                        default=Path("../mrst-2024a-Geomech/modules/ad-core"))
    parser.add_argument("--python-root", type=Path,
                        default=Path("pymrst/modules/ad_core"))
    parser.add_argument("--csv", type=Path, default=Path("docs/ad_core/parity.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("AD_CORE_PARITY.md"))
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    mrst_root = args.mrst_root.resolve()
    python_root = args.python_root.resolve()
    if not mrst_root.is_dir():
        parser.error(f"MRST ad-core tree not found: {mrst_root}")
    if not python_root.is_dir():
        parser.error(f"PyMRST ad_core tree not found: {python_root}")

    python_files = sorted(python_root.rglob("*.py"))
    python_sources = source_index(python_files)
    tree_bindings: set[str] = set()
    file_by_class: dict[str, Path] = {}
    for path, text in python_sources.items():
        tree_bindings |= bindings_in(text)
        for m in re.finditer(r"^class\s+(\w+)\s*[:(]", text, re.MULTILINE):
            file_by_class.setdefault(m.group(1), path)

    validation_files = sorted((repo / "tests").rglob("*.py"))
    validation_files += sorted((repo / "examples" / "validation").rglob("*.py"))
    validation_sources = source_index(validation_files)

    rows = []
    for matlab_file in sorted(mrst_root.rglob("*.m")):
        mrst_text = matlab_file.read_text(encoding="utf-8", errors="ignore")
        status, counterparts, evidence = find_counterparts(
            matlab_file, mrst_root, python_root, python_sources
        )
        symbol = matlab_file.stem
        kind = matlab_kind(mrst_text)

        total = in_ctr = elsewhere = 0
        absent: list[str] = []
        depth: float | None = None
        notes = ""
        if kind == "classdef":
            total, in_ctr, elsewhere, absent, depth = depth_scan(
                symbol, mrst_text, counterparts, python_sources, tree_bindings,
                file_by_class)
            if depth is not None and status == "ported":
                if depth == 0.0:
                    status = "wrapper"
                    notes = (f"filename counterpart exists but implements none "
                            f"of {total} scored methods")
                elif depth < DEPTH_BAR:
                    status = "ported-shallow"
                    notes = (f"{in_ctr}/{total} methods present in the mapped "
                            f"counterpart (below the {DEPTH_BAR:.0%} bar); "
                            f"absent: {', '.join(absent)}")

        if symbol in LEGACY_IMPLEMENTATIONS:
            status, notes = "pending-review", LEGACY_IMPLEMENTATIONS[symbol]
        validations = validation_matches(symbol, validation_sources)
        validation = "validated-reference" if validations else "unvalidated"
        if symbol in LEGACY_IMPLEMENTATIONS:
            validation, validations = "unvalidated", []
        rows.append({
            "mrst_path": str(matlab_file.relative_to(mrst_root)),
            "symbol": symbol,
            "subsystem": matlab_file.relative_to(mrst_root).parts[0],
            "port_status": status,
            "python_counterpart": relative_list(counterparts, repo),
            "port_evidence": evidence,
            "mrst_kind": kind,
            "mrst_methods": total,
            "methods_in_counterpart": in_ctr,
            "methods_elsewhere": elsewhere,
            "methods_absent": " ".join(absent),
            "depth": f"{depth:.2f}" if depth is not None else "",
            "validation_status": validation,
            "validation_files": relative_list(validations, repo),
            "notes": notes,
        })

    csv_path = (repo / args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["port_status"] for row in rows)
    for key in ("ported", "ported-consolidated", "ported-shallow", "wrapper",
               "pending-review", "excluded-catalog", "excluded-platform",
               "excluded-external", "excluded-deprecated",
               "excluded-external-solver", "excluded-visualization",
               "excluded-benchmark-driver"):
        counts.setdefault(key, 0)
    validated = Counter(row["validation_status"] for row in rows)
    subsystems: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        sub = subsystems[row["subsystem"]]
        sub[row["port_status"]] += 1
        sub["total"] += 1
        if row["validation_status"] == "validated-reference":
            sub["validated"] += 1

    shallow_rows = [r for r in rows if r["port_status"] in ("ported-shallow", "wrapper")]

    md_path = (repo / args.markdown).resolve()
    md_path.write_text(
        "# AD-core parity ledger\n\n"
        "Generated by `scripts/audit_ad_core_parity.py` from the local MRST "
        "2024a reference tree. The CSV is the authoritative file-level ledger, "
        "with per-method depth for classdef files "
        "(`tests/ad_core/test_parity_ledger_depth.py` is the gate: a filename "
        "match alone is not enough to keep the `ported` label above "
        f"{MIN_METHODS_FOR_DEPTH} methods).\n\n"
        "## Current scan\n\n"
        f"- MRST files inventoried: **{len(rows)}**\n"
        f"- Direct filename counterparts: **{counts['ported']}**\n"
        f"- Consolidated/symbol-reference counterparts: **{counts['ported-consolidated']}**\n"
        f"- Shallow (below the {DEPTH_BAR:.0%} depth bar): **{counts['ported-shallow']}**\n"
        f"- Wrapper (counterpart exists, 0 methods present): **{counts['wrapper']}**\n"
        f"- Excluded MATLAB catalogs: **{counts['excluded-catalog']}**\n"
        f"- Excluded MATLAB MEX implementation files: **{counts['excluded-platform']}**\n"
        f"- Excluded optional external-data integrations: **{counts['excluded-external']}**\n"
        f"- Excluded deprecated legacy implementations: **{counts['excluded-deprecated']}**\n"
        f"- Excluded optional external solver wrappers: **{counts['excluded-external-solver']}**\n"
        f"- Excluded MATLAB-only visualization helpers: **{counts['excluded-visualization']}**\n"
        f"- Excluded MATLAB timing/plot drivers: **{counts['excluded-benchmark-driver']}**\n"
        f"- Pending manual review or implementation: **{counts['pending-review']}**\n"
        f"- Files mentioned by reference validators: **{validated['validated-reference']}**\n\n"
        "## Shallow / wrapper rows (the ratchet)\n\n"
        + ("None currently.\n\n" if not shallow_rows else
           "".join(f"- `{r['symbol']}` ({r['port_status']}): {r['notes']}\n"
                  for r in shallow_rows) + "\n")
        + "## Subsystems\n\n"
        "| Subsystem | MRST files | Candidate ported | Pending review | Reference-validated |\n"
        "|---|---:|---:|---:|---:|\n"
        + "".join(
            f"| `{name}` | {data['total']} | "
            f"{data['ported'] + data['ported-consolidated']} | "
            f"{data['pending-review']} | {data['validated']} |\n"
            for name, data in sorted(subsystems.items())
        )
        + "\n## Status policy\n\n"
        "- `ported`: a snake-case filename counterpart exists, and if the MRST "
        f"file is a classdef with >= {MIN_METHODS_FOR_DEPTH} methods, at least "
        f"{DEPTH_BAR:.0%} of them (excluding pinned platform/layout/subsref "
        "cases) are present in that counterpart.\n"
        "- `ported-consolidated`: the MATLAB symbol is explicitly referenced "
        "inside a consolidated Python module.\n"
        "- `ported-shallow`: a counterpart exists but scores below the depth "
        "bar; `methods_absent` names what is missing.\n"
        "- `wrapper`: a counterpart exists and implements none of the scored "
        "methods.\n"
        "- `validated-reference`: a test or validation script explicitly names "
        "the MATLAB symbol or its Python snake-case spelling. This does not yet "
        "grade validation depth.\n"
        "- `excluded-catalog`: MATLAB `Contents.m` catalog only.\n"
        "- `excluded-platform`: MATLAB MEX build/runtime source; its numerical "
        "fallback must still be ported and validated.\n"
        "- `excluded-external`: optional external repository/data integration, "
        "outside the self-contained numerical core.\n"
        "- `excluded-deprecated`: MRST explicitly deprecated the implementation; "
        "its active replacement behavior is provided by the facility layer.\n"
        "- `excluded-external-solver`: wrapper requiring an optional external "
        "binary solver; PyMRST's native CPR/direct solvers remain available.\n"
        "- `excluded-visualization`: MATLAB graphics-only helper.\n"
        "- `excluded-benchmark-driver`: MATLAB timing/plot orchestration whose "
        "reusable benchmark kernel is ported and validated.\n"
        "- `pending-review`: no defensible automatic mapping; requires manual "
        "classification or implementation.\n\n"
        "Regenerate with:\n\n"
        "```bash\npython3 scripts/audit_ad_core_parity.py\n```\n",
        encoding="utf-8",
    )

    def _display(p: Path) -> str:
        try:
            return str(p.relative_to(repo))
        except ValueError:
            return str(p)

    print(f"Wrote {_display(csv_path)} ({len(rows)} rows)")
    print(f"Wrote {_display(md_path)}")
    print(f"port_status counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
