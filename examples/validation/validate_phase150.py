"""Phase 150: exact compositional example catalog parity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.contents import (  # noqa: E402
    EXAMPLE_PORTS,
    list_example_ports,
)


MRST_EXAMPLES = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech/modules/compositional/examples")
PYMRST_EXAMPLES = ROOT / "pymrst" / "modules" / "compositional" / "examples"


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mrst_example_files() -> set[str]:
    return {
        path.relative_to(MRST_EXAMPLES).as_posix()
        for path in MRST_EXAMPLES.rglob("*.m")
    }


def validate_catalog_covers_mrst_tree() -> int:
    mrst_files = _mrst_example_files()
    catalog_files = set(EXAMPLE_PORTS)
    failures = 0
    failures += _check("MRST compositional example tree is available",
                       MRST_EXAMPLES.is_dir() and bool(mrst_files))
    failures += _check("catalog has one entry per MRST .m example",
                       catalog_files == mrst_files)
    if catalog_files != mrst_files:
        print(f"    missing from catalog: {sorted(mrst_files - catalog_files)}")
        print(f"    stale catalog entries: {sorted(catalog_files - mrst_files)}")
    return failures


def validate_registered_pymrst_ports_exist_and_import() -> int:
    failures = 0
    ports = list_example_ports()
    failures += _check("list_example_ports returns a defensive copy",
                       ports is not EXAMPLE_PORTS
                       and ports["Contents.m"] is not EXAMPLE_PORTS["Contents.m"])
    for mrst_file, meta in sorted(ports.items()):
        py_file = PYMRST_EXAMPLES / meta["file"]
        failures += _check(f"{mrst_file} has a PyMRST file",
                           py_file.is_file())
        try:
            importlib.import_module(meta["python"])
        except Exception as exc:  # pragma: no cover - printed for diagnostics
            print(f"    import failed for {meta['python']}: {exc}")
            failures += _check(f"{mrst_file} imports", False)
        else:
            failures += _check(f"{mrst_file} imports", True)
    return failures


def main() -> int:
    print("\nPhase 150: exact compositional example catalog parity")
    failures = validate_catalog_covers_mrst_tree()
    failures += validate_registered_pymrst_ports_exist_and_import()
    if failures:
        print(f"\nPhase 150 failed ({failures} checks).")
        return 1
    print("\nEvery MRST compositional example now has a registered PyMRST port.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
