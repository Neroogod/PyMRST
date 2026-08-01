"""Programmatic SPE9 tutorial workflow without mandatory plotting."""

from pymrst.modules.ad_blackoil.examples.spe9.simulate_spe9 import setup_spe9, simulate_spe9


def run_blackoil_tutorial_spe9(nsteps=None, *, deck_path=None, verbose=True):
    """Set up, simulate, and return reservoir states plus summary vectors."""
    return simulate_spe9(nsteps, deck_path=deck_path, verbose=verbose)


__all__ = ["setup_spe9", "run_blackoil_tutorial_spe9"]


if __name__ == "__main__":
    run_blackoil_tutorial_spe9()
