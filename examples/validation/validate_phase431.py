"""Phase 431: compare readSummaryLocal with MRST on SPE9 output."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.utils import read_summary_local

HERE = Path(__file__).resolve().parent
REF = HERE / "ref_p431_read_summary.mat"
PREFIX = (
    HERE.parents[2]
    / "mrst-2024a-Geomech/modules/ad-blackoil/examples/spe9/compare/SPE9"
)


def _cells(value):
    return [str(item.item()).strip() for item in np.asarray(value).ravel()]


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase431_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    summary, _ = read_summary_local(PREFIX, ["TIME", "WBHP", "WOPR"])
    sample_name = _cells(ref["out_sample_name"])[0]

    actual = (
        ("names", summary.WGNAMES, _cells(ref["out_names"])),
        ("keywords", summary.KEYWORDS, _cells(ref["out_keywords"])),
        ("name map", summary.nInx + 1, ref["out_ninx"].ravel()),
        ("keyword map", summary.kInx + 1, ref["out_kinx"].ravel()),
        ("data", summary.data, ref["out_data"]),
        ("WBHP names", summary.get_names("WBHP"), _cells(ref["out_wbhp_names"])),
        ("sample keywords", summary.get_keywords(sample_name),
         _cells(ref["out_sample_keywords"])),
        ("sample WBHP", summary.get(sample_name, "WBHP"),
         ref["out_sample_wbhp"]),
    )
    failed = 0
    for name, value, expected in actual:
        if isinstance(value, list):
            ok = value == expected
            detail = f"{len(value)} entries"
        else:
            value = np.asarray(value)
            expected = np.asarray(expected)
            error = np.max(np.abs(value - expected)) if value.size else 0.0
            ok = value.shape == expected.shape and error < 2e-6
            detail = f"shape={value.shape}, max|d|={error:.3e}"
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    print("\nreadSummaryLocal matches MRST!" if not failed
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
