"""Compatibility entry point for MRST ``spe1/simulateSPE1.m``.

The real SPE1 tutorial implementation lives in
``blackoil_tutorial_spe1.py``. This file exists so the MRST-shaped
``simulate_spe1`` import path remains available.
"""

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    SPE1TutorialResult,
    print_tutorial_summary,
    run_blackoil_tutorial_spe1,
    setup_spe1_tutorial,
)


def simulate_spe1(nsteps: int | None = 30) -> SPE1TutorialResult:
    return run_blackoil_tutorial_spe1(nsteps=nsteps)


if __name__ == "__main__":
    print_tutorial_summary(simulate_spe1(30))


__all__ = ["SPE1TutorialResult", "setup_spe1_tutorial", "simulate_spe1"]
