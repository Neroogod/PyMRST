"""Credential-free CLI used by exported SSH/Slurm experiment bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ensemble_jobs import MEMBER_RUNNERS, execute_ensemble


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute a sealed PyMRST experiment matrix")
    parser.add_argument("--materialized", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", choices=sorted(MEMBER_RUNNERS), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-per-worker", type=int, default=512*1024*1024)
    values = parser.parse_args(argv)
    result = execute_ensemble(
        Path(values.materialized), Path(values.output), runner_kind=values.runner,
        max_workers=values.workers,
        memory_per_worker_bytes=values.memory_per_worker,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
