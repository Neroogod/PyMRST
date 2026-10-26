"""Structured multidimensional interpolation with AD chain rules.

MRST's `getMultiDimInterpolator.m` is ported in full in
`get_multi_dim_interpolator.py`; this module keeps the earlier import path.
"""

from pymrst.modules.ad_core.utils.get_multi_dim_interpolator import (
    getMultiDimInterpolator,
    get_multi_dim_interpolator,
)

__all__ = ["getMultiDimInterpolator", "get_multi_dim_interpolator"]


if __name__ == "__main__":
    def test_the_import_path_reaches_the_port():
        import importlib
        module = importlib.import_module("pymrst.modules.ad_core.utils.get_multi_dim_interpolator")
        assert get_multi_dim_interpolator is module.get_multi_dim_interpolator
        print("ok test_the_import_path_reaches_the_port")

    test_the_import_path_reaches_the_port()
    print("\nAll tests passed!")
