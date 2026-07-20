"""Port of MRST ``blackoilTimeIntegrationExample.m``.

The MRST example compares fully implicit, explicit, and adaptive-implicit
backends for SPE1/SPE9. PyMRST currently runs the fully implicit SPE1 path and
records the same state/well histories; explicit/AIM backends are not present.
"""

from pymrst.modules.ad_blackoil.examples.spe1.simulate_spe1 import simulate_spe1


def run_blackoil_time_integration_example(name: str = "spe1", nsteps: int = 10):
    if name.lower() != "spe1":
        raise NotImplementedError("PyMRST only has the SPE1 deck reader for this example")
    return {"Fully-Implicit": simulate_spe1(nsteps)}


if __name__ == "__main__":
    out = run_blackoil_time_integration_example("spe1", 5)
    print(f"black-oil time integration variants: {list(out)}")
