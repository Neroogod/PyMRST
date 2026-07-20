"""Layout port of MRST ``spe9/blackoilTutorialSPE9.m``."""

from pymrst.modules.ad_blackoil.examples.spe9.simulate_spe9 import simulate_spe9


def run_blackoil_tutorial_spe9():
    return simulate_spe9()


if __name__ == "__main__":
    run_blackoil_tutorial_spe9()
