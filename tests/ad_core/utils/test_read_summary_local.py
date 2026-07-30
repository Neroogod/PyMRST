from pathlib import Path

import numpy as np

from pymrst.modules.ad_core.utils import read_summary_local


PREFIX = (
    Path(__file__).parents[4]
    / "mrst-2024a-Geomech/modules/ad-blackoil/examples/spe9/compare/SPE9"
)


def test_spe9_summary_filter_and_accessors():
    summary, spec = read_summary_local(PREFIX, ["TIME", "WBHP", "WOPR"])
    assert "WGNAMES" in spec and "KEYWORDS" in spec
    assert summary.data.shape[0] == summary.nInx.size == summary.kInx.size
    assert summary.data.shape[1] > 20
    assert set(summary.KEYWORDS) == {"TIME", "WBHP", "WOPR"}
    assert np.all(np.diff(summary.get(keyword="TIME").ravel()) >= 0)

    wells = summary.get_names("WBHP")
    assert wells
    for well in wells:
        assert "WBHP" in summary.get_keywords(well)
        assert summary.get(well, "WBHP").shape[1] == summary.data.shape[1]
