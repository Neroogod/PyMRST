#!/usr/bin/env python3
"""Run the reproducible PyMRST shale qualification gate.

The default gate is headless and self-contained.  ``--matlab`` first
regenerates all deterministic MRST fixtures with the installed MATLAB, then
runs the same Python comparisons against those fresh files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MATLAB_GENERATORS = (
    ("dump_shale_foundation_reference", "foundation.mat"),
    ("dump_pedfm_projection_reference", "pedfm_projection.mat"),
    ("dump_shale_spe_geometry_reference", "spe_geometry.mat"),
    ("dump_shale_fluid_catalog_reference", "fluid_catalog.mat"),
    ("dump_shale_operator_pipeline_reference", "operator_pipeline.mat"),
)


def run(command, *, cwd: Path, env=None):
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab", action="store_true",
                        help="regenerate all direct MRST fixtures first")
    parser.add_argument("--matlab-bin", type=Path,
                        default=Path("/Applications/MATLAB_R2021b.app/bin/matlab"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/shale/qualification.json"))
    parser.add_argument("--pytest-args", nargs="*", default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = []

    if args.matlab:
        if not args.matlab_bin.is_file():
            parser.error(f"MATLAB executable not found: {args.matlab_bin}")
        fixture_dir = root / "tests/fixtures/shale"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        matlab_scripts = (root / "scripts/matlab/shale").as_posix().replace("'", "''")
        targets = [fixture_dir / filename for _, filename in MATLAB_GENERATORS]
        before = {target: target.stat().st_mtime_ns if target.exists() else -1
                  for target in targets}
        calls = []
        for function, filename in MATLAB_GENERATORS:
            output = (fixture_dir / filename).as_posix().replace("'", "''")
            calls.append(f"{function}('{output}')")
        expression = f"addpath('{matlab_scripts}'); " + "; ".join(calls)
        matlab_record = run([str(args.matlab_bin), "-batch", expression], cwd=root)
        generated = all(
            target.is_file() and target.stat().st_mtime_ns > before[target]
            for target in targets
        )
        # MATLAB R2021b can abort while shutting down on newer macOS versions
        # after successfully writing every fixture. Preserve its raw exit code,
        # but let the numerical comparisons decide fixture validity in that
        # narrowly identifiable case.
        matlab_record["process_returncode"] = matlab_record["returncode"]
        matlab_record["fixtures_generated"] = generated
        if generated:
            matlab_record["returncode"] = 0
        records.append(matlab_record)

    if not records or records[-1]["returncode"] == 0:
        records.append(run([sys.executable, "scripts/audit_shale_parity.py"], cwd=root))
    if records[-1]["returncode"] == 0:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("MPLCONFIGDIR", "/tmp/pymrst-shale-mpl")
        records.append(run(
            [sys.executable, "-m", "pytest", "-q", "tests/shale", *args.pytest_args],
            cwd=root, env=env,
        ))

    result = {
        "schema": "pymrst.shale.qualification.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matlab_regenerated": bool(args.matlab),
        "passed": all(item["returncode"] == 0 for item in records),
        "steps": records,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    for item in records:
        print(f"[{item['returncode']}] {item['seconds']:.2f}s {' '.join(item['command'])}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
