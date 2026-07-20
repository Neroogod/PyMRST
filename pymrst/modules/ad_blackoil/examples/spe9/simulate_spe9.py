"""Layout port of MRST ``spe9/simulateSPE9.m``."""

from pymrst.modules.ad_blackoil.examples._unsupported import unsupported_example


def simulate_spe9():
    unsupported_example(
        "simulateSPE9",
        "SPE9 deck parsing, VFP/deck schedule import, and summary comparison "
        "are not present in PyMRST",
    )


if __name__ == "__main__":
    simulate_spe9()
