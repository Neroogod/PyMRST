#!/usr/bin/env python3
"""Merge independently recorded Norne cases into one acceptance manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    (fluid, method)
    for fluid in ("water", "oil water", "blackoil")
    for method in ("fully coupled", "fixed stress splitting")
}


def merge(paths: list[Path]) -> dict:
    runs = {}
    sources = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        for run in payload.get("runs", []):
            runs[(run["fluid_model"], run["method"])] = run
    missing = EXPECTED - set(runs)
    failed = {key: run.get("error", "failed") for key, run in runs.items()
              if key in EXPECTED and not run.get("converged", False)}
    if missing or failed:
        raise RuntimeError(f"acceptance incomplete: missing={sorted(missing)}, failed={failed}")
    ordered = [runs[(fluid, method)] for fluid in ("water", "oil water", "blackoil")
               for method in ("fully coupled", "fixed stress splitting")]
    fingerprints = {run.get("grid_rock_sha256") for run in ordered}
    if None in fingerprints or len(fingerprints) != 1:
        raise RuntimeError(f"grid/rock fingerprints differ: {sorted(map(str, fingerprints))}")
    if any(run.get("cells") != 7392 or run.get("report_steps") != 21 for run in ordered):
        raise RuntimeError("every full-field case must contain 7,392 cells and 21 report steps")
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "norne_case": "full",
        "bc_case": "bottom fixed",
        "converged": True,
        "grid_rock_sha256": next(iter(fingerprints)),
        "sources": sources,
        "runs": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote six-case acceptance manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
