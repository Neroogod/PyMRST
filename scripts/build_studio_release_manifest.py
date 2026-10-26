#!/usr/bin/env python3
"""Create and optionally sign a PyMRST Studio release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymrst.app.release import create_release_manifest, sign_manifest, write_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path)
    arguments = parser.parse_args(argv)
    manifest = create_release_manifest(
        arguments.root, arguments.artifact, version=arguments.version,
        platform=arguments.platform, architecture=arguments.architecture,
    )
    write_manifest(manifest, arguments.output)
    result = {"manifest": str(arguments.output), "content_hash": manifest.content_hash}
    if arguments.signing_key:
        signature = sign_manifest(manifest, arguments.signing_key.read_bytes())
        signature_path = arguments.output.with_suffix(arguments.output.suffix + ".sig.json")
        signature_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
        result["signature"] = str(signature_path)
        result["key_id"] = signature["key_id"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
