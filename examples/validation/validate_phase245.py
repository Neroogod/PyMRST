"""Phase 245: separator facility Contents.m catalog parity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators.contents import (  # noqa: E402
    MRST_CONTENTS_SEPARATORS,
    SEPARATORS,
    list_mrst_contents_separators,
    list_separators,
)


EXPECTED = (
    ("BasicSeparator", ""),
    ("EOSSeparator", ""),
    ("SeparatorGroup", ""),
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    listed = list_mrst_contents_separators()
    catalog = list_separators()

    print("\nPhase 245: separator facility Contents.m catalog parity")
    failures = 0
    failures += _check("separator Contents.m entries match MRST order/text",
                       listed == EXPECTED)
    failures += _check("separator Contents constant is immutable tuple metadata",
                       isinstance(MRST_CONTENTS_SEPARATORS, tuple)
                       and all(isinstance(item, tuple) for item in MRST_CONTENTS_SEPARATORS))
    failures += _check("helpers return defensive copies",
                       listed == MRST_CONTENTS_SEPARATORS
                       and listed is not MRST_CONTENTS_SEPARATORS
                       and catalog is not SEPARATORS
                       and catalog["EOSSeparator"] is not SEPARATORS["EOSSeparator"])
    failures += _check("all separator Contents entries are represented in catalog",
                       set(catalog) == {name for name, _ in listed}
                       and all(meta["status"] == "implemented" for meta in catalog.values()))
    failures += _check("implemented separator modules import",
                       all(importlib.import_module(
                           f"pymrst.modules.ad_core.models.facilities.separators.{meta['python_module']}"
                       ) is not None for meta in catalog.values()))

    if failures:
        print(f"\nPhase 245 failed ({failures} checks).")
        return 1
    print("\nSeparator facility catalog now mirrors MRST Contents.m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
