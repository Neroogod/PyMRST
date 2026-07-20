"""Layout port of MRST ``fieldModelNorneExample.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def run_field_model_norne_example():
    unsupported_example(
        "fieldModelNorneExample",
        "the full Norne black-oil deck/schedule import is not ported; "
        "PyMRST's geomechanics Norne example lives in top-level examples",
    )


if __name__ == "__main__":
    run_field_model_norne_example()
