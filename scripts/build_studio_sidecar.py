#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymrst.app.packaging import build_sidecar, native_target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the native managed PyMRST sidecar")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", default=native_target())
    parser.add_argument("--python", default="")
    arguments = parser.parse_args(argv)
    keywords = {"target": arguments.target}
    if arguments.python:
        keywords["python_executable"] = arguments.python
    print(json.dumps(build_sidecar(arguments.root, **keywords).to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
