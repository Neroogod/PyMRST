#!/usr/bin/env python3
"""Generate a conservative MRST-shale/PyMRST owner and asset ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path


OWNER_MAP = {
    "Fig10to13TeneCaseLowKf": "spe201243pa/fig_10_to_13_tene_low_kf.py",
    "Fig10to13TeneCasesHighKf": "spe201243pa/fig_10_to_13_tene_high_kf.py",
    "Fig15_17_18_validationCase2": "spe201243pa/fig_15_17_18_validation_case_2.py",
    "Fig16_validationCase2_structured": "spe201243pa/fig_16_validation_case_2_structured.py",
    "Fig21_22_23_bakken_edfm": "spe201243pa/fig_21_22_23_bakken_edfm.py",
    "Fig21_22_23_bakken_pedfm": "spe201243pa/fig_21_22_23_bakken_pedfm.py",
    "StructuredGridFractureonFace": "spe201243pa/structured_grid_fracture_on_face.py",
    "explicitStencil3Fracs": "spe201243pa/explicit_stencil_3_fracs.py",
    "compositional3D": "examples/compositional_3d.py",
    "eagleFordEDFM_MEX": "examples/eagle_ford_edfm_accelerated.py",
    "eagleFordEDFMfull": "examples/eagle_ford_edfm_full.py",
    "eagleFord_MEX": "examples/eagle_ford_accelerated.py",
    "eagleFordfull": "examples/eagle_ford_full.py",
    "stochasticFracs": "examples/stochastic_fractures.py",
    "EDFMshalegrid": "hfm_fns/edfm_shale_grid.py",
    "setupShaleEDFMOpsTPFA": "hfm_fns/setup_shale_edfm_operators_tpfa.py",
    "wellfractureNNCs3D": "other_mrst_fns/well_fracture_nncs_3d.py",
    "setupPEDFMOpsTPFA": "pedfm_nnc/setup_pedfm_operators_tpfa.py",
    "transmultpEDFM": "pedfm_nnc/transmissibility_multiplier_pedfm.py",
    "Pts2CellIdx": "utils/points_to_cell_indices.py",
    "createMultiStageHFs": "utils/create_multistage_hfs.py",
    "gangiFn": "utils/gangi.py",
    "meshHFsystem": "utils/mesh_hf_system.py",
    "plotfracSystem_NF": "utils/plot_frac_system_nf.py",
    "processStochFracs": "utils/process_stochastic_fractures.py",
    "eqnsShaleNaturalVars": "compositional_fns/equations_shale_natural_vars.py",
    "fracturefractureShaleNNCs3D": "hfm_fns/fracture_fracture_shale_nncs_3d.py",
    "fracturematrixShaleNNC3D": "hfm_fns/fracture_matrix_shale_nncs_3d.py",
    "pMatFracNNCs3D": "pedfm_nnc/projected_matrix_fracture_nncs_3d.py",
    "MulistagePlanarNonPlanarHF": "spe201243pa/multistage_planar_nonplanar_hf.py",
}


def snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_owner(matlab_file: Path, mrst_root: Path) -> Path | None:
    rel = matlab_file.relative_to(mrst_root)
    if matlab_file.stem == "Contents":
        return Path("examples/contents.py")
    mapped = OWNER_MAP.get(matlab_file.stem)
    if mapped:
        return Path(mapped)
    directories = {
        "compositionalFns": "compositional_fns",
        "hfmFns": "hfm_fns",
        "otherMRSTfns": "other_mrst_fns",
        "pEDFM_nnc": "pedfm_nnc",
        "SPE201243PA": "spe201243pa",
    }
    parent = directories.get(rel.parts[0], rel.parts[0].lower())
    return Path(parent) / f"{snake_case(matlab_file.stem)}.py"


def mentioned_in_tests(symbol: str, expected: Path, tests: dict[Path, str]) -> list[Path]:
    names = {symbol.lower(), snake_case(symbol), expected.stem}
    return [
        path for path, text in tests.items()
        if any(re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text, re.I)
               for name in names)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrst-root", type=Path,
                        default=Path("../mrst-2024a-Geomech/modules/shale"))
    parser.add_argument("--python-root", type=Path,
                        default=Path("pymrst/modules/shale"))
    parser.add_argument("--csv", type=Path, default=Path("docs/shale/parity.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("SHALE_PARITY_AUDIT.md"))
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    mrst_root, python_root = args.mrst_root.resolve(), args.python_root.resolve()
    if not mrst_root.is_dir() or not python_root.is_dir():
        parser.error("run from PyMRST with both MRST and PyMRST shale trees present")

    test_sources = {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted((repo / "tests" / "shale").rglob("*.py"))
    }
    rows = []
    for source in sorted(mrst_root.rglob("*.m")):
        expected = expected_owner(source, mrst_root)
        target = python_root / expected
        evidence = mentioned_in_tests(source.stem, expected, test_sources)
        rows.append({
            "kind": "owner",
            "mrst_path": str(source.relative_to(mrst_root)),
            "python_path": str(target.relative_to(repo)),
            "layout_status": "present" if target.is_file() else "missing",
            "content_status": "catalog" if source.stem == "Contents" else "native-port",
            "validation_status": "python-tested" if evidence else "unreferenced-by-test",
            "validation_files": "; ".join(str(p.relative_to(repo)) for p in evidence),
            "sha256_match": "not-applicable",
        })
    for source in sorted(p for p in mrst_root.rglob("*") if p.suffix.lower() in {".mat", ".csv"}):
        rel = source.relative_to(mrst_root)
        relative_asset = (
            Path("spe201243pa/data") / source.name
            if rel.parts[0] == "SPE201243PA"
            else rel
        )
        target = python_root / relative_asset
        rows.append({
            "kind": "asset",
            "mrst_path": str(rel),
            "python_path": str(target.relative_to(repo)),
            "layout_status": "present" if target.is_file() else "missing",
            "content_status": "bundled-data",
            "validation_status": "byte-certified" if target.is_file() and sha256(source) == sha256(target) else "unvalidated",
            "validation_files": "",
            "sha256_match": str(target.is_file() and sha256(source) == sha256(target)).lower(),
        })

    csv_path = (repo / args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["kind"], row["layout_status"]) for row in rows)
    validations = Counter(row["validation_status"] for row in rows)
    missing = [row for row in rows if row["layout_status"] == "missing"]
    md_path = (repo / args.markdown).resolve()
    md_path.write_text(
        "# Shale parity audit\n\n"
        "Generated by `scripts/audit_shale_parity.py`. Presence proves layout "
        "coverage only; numerical parity requires the separate Python and MATLAB "
        "qualification evidence recorded in `shale_parity.md`.\n\n"
        f"- MRST owners: **{sum(row['kind'] == 'owner' for row in rows)}**\n"
        f"- Owners present in MRST-shaped Python layout: **{counts[('owner', 'present')]}**\n"
        f"- Owners referenced by shale tests: **{validations['python-tested']}**\n"
        f"- MRST data assets: **{sum(row['kind'] == 'asset' for row in rows)}**\n"
        f"- Byte-certified data assets: **{validations['byte-certified']}**\n"
        f"- Missing layout entries: **{len(missing)}**\n\n"
        "The machine-readable ledger is `docs/shale/parity.csv`.\n",
        encoding="utf-8",
    )
    print(f"wrote {csv_path.relative_to(repo)} and {md_path.relative_to(repo)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
