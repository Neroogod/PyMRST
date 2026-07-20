"""Layout port of MRST ``egg/eggExample.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def run_egg_example():
    unsupported_example(
        "eggExample",
        "the EGG benchmark dataset importer is not present in PyMRST",
    )


if __name__ == "__main__":
    run_egg_example()
