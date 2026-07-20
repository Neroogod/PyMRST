"""Layout port of MRST ``spe3/simulateSPE3.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def simulate_spe3():
    unsupported_example(
        "simulateSPE3",
        "SPE3 deck parsing and black-oil schedule import are not present in PyMRST",
    )


if __name__ == "__main__":
    simulate_spe3()
