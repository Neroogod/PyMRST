"""MRST ``terzaghi.m`` compatibility example.

PyMRST validates Terzaghi consolidation through the 3D VEM poromechanics
implementation in ``pymrst.modules.ad_mechanics.poromech``. This module gives
that benchmark an MRST-shaped landing file.
"""

from pymrst.modules.ad_mechanics.poromech import MechWaterModel


def run_terzaghi():
    """Run the built-in poromechanics self-test that includes Terzaghi physics."""
    # The full analytic checks live in poromech's module self-test to avoid
    # duplicating the benchmark setup in two places.
    from pymrst.modules.ad_mechanics import poromech

    return poromech


__all__ = ["run_terzaghi", "MechWaterModel"]


if __name__ == "__main__":
    run_terzaghi()
    print("ok terzaghi compatibility example")
