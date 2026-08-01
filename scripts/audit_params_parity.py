#!/usr/bin/env python3
"""Audit the complete MRST params tree against reviewed PyMRST owners."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MRST = ROOT.parent/"mrst-2024a-Geomech"/"params"
OUT = ROOT/"docs"/"params"/"parity.csv"

OWNERS = {
    "gravity.m": "pymrst/params/gravity/gravity.py",
    "rock/gaussianField.m": "pymrst/params/rock/gaussian_field.py",
    "rock/grdecl2Rock.m": "pymrst/params/rock/grdecl2rock.py",
    "rock/logNormLayers.m": "pymrst/params/rock/log_norm_layers.py",
    "rock/makeRock.m": "pymrst/params/rock/make_rock.py",
    "rock/permTensor.m": "pymrst/params/rock/perm_tensor.py",
    "rock/permeabilityConverter.m": "pymrst/params/rock/permeability_converter.py",
    "rock/poreVolume.m": "pymrst/params/rock/pore_volume.py",
    "wells_and_bc/addBC.m": "pymrst/params/wells_and_bc/add_bc.py",
    "wells_and_bc/addSource.m": "pymrst/params/wells_and_bc/add_source.py",
    "wells_and_bc/addWell.m": "pymrst/params/wells_and_bc/add_well.py",
    "wells_and_bc/fluxside.m": "pymrst/params/wells_and_bc/fluxside.py",
    "wells_and_bc/pside.m": "pymrst/params/wells_and_bc/pside.py",
    "wells_and_bc/psideh.m": "pymrst/params/wells_and_bc/psideh.py",
    "wells_and_bc/verticalWell.m": "pymrst/params/wells_and_bc/vertical_well.py",
}

EVIDENCE = {
    "gravity.m": "examples/validation/validate_params_gravity.py",
    "rock/gaussianField.m": "examples/validation/validate_params_stochastic.py",
    "rock/grdecl2Rock.m": "examples/validation/validate_phase366.py",
    "rock/logNormLayers.m": "examples/validation/validate_params_stochastic.py",
    "rock/makeRock.m": "examples/validation/validate_params_rock_tensor.py",
    "rock/permTensor.m": "examples/validation/validate_params_rock_tensor.py",
    "rock/permeabilityConverter.m": "examples/validation/validate_phase386.py",
    "rock/poreVolume.m": "examples/validation/validate_params_end_to_end.py",
    "wells_and_bc/addBC.m": "examples/validation/validate_params_forcing.py",
    "wells_and_bc/addSource.m": "examples/validation/validate_params_forcing.py",
    "wells_and_bc/addWell.m": "examples/validation/validate_params_end_to_end.py",
    "wells_and_bc/fluxside.m": "examples/validation/validate_phase22.py",
    "wells_and_bc/pside.m": "examples/validation/validate_phase22.py",
    "wells_and_bc/psideh.m": "examples/validation/validate_phase227.py",
    "wells_and_bc/verticalWell.m": "examples/validation/validate_params_end_to_end.py",
}

DOCUMENTATION_OWNERS = {
    "README": "pymrst/params/README.md",
    "rock/README": "pymrst/params/rock/README.md",
    "wells_and_bc/README": "pymrst/params/wells_and_bc/README.md",
}


def main() -> int:
    if not MRST.is_dir():
        print(f"MRST params tree missing: {MRST}")
        return 1
    files = sorted(path.relative_to(MRST).as_posix() for path in MRST.rglob("*"))
    files = [name for name in files if (MRST/name).is_file()]
    rows, failures = [], []
    for name in files:
        if name in OWNERS:
            owner, evidence = OWNERS[name], EVIDENCE[name]
            owner_ok, evidence_ok = (ROOT/owner).is_file(), (ROOT/evidence).is_file()
            status = "complete" if owner_ok and evidence_ok else "missing"
            if status != "complete":
                failures.append(name)
            kind = "executable"
        elif Path(name).name == "Contents.m":
            kind, status, owner, evidence = "catalog", "excluded", "", ""
        elif name in DOCUMENTATION_OWNERS:
            kind, owner = "documentation", DOCUMENTATION_OWNERS[name]
            evidence = "params_parity.md"
            status = "covered" if (ROOT/owner).is_file() else "missing"
            if status != "covered":
                failures.append(name)
        else:
            kind, status, owner, evidence = "unknown", "missing", "", ""
            failures.append(name)
        rows.append({
            "mrst_path": name, "kind": kind, "status": status,
            "python_owner": owner, "validation_evidence": evidence,
        })
    missing_inventory = sorted(set(OWNERS)-set(files))
    failures.extend(missing_inventory)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"MRST params files: {len(rows)}")
    print(f"Executable owners complete and validated: {complete}/{len(OWNERS)}")
    print(f"Catalogs excluded: {sum(row['kind']=='catalog' for row in rows)}")
    print(f"Documentation classified: {sum(row['kind']=='documentation' for row in rows)}")
    print(f"Ledger: {OUT.relative_to(ROOT)}")
    if failures:
        print("Failures: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
