"""SPE1 ad-blackoil examples."""

__all__ = ["simulate_spe1"]


def __getattr__(name):
    if name == "simulate_spe1":
        from pymrst.modules.ad_blackoil.examples.spe1.simulate_spe1 import simulate_spe1
        return simulate_spe1
    raise AttributeError(name)
