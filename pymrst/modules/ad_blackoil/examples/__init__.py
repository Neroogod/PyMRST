"""Python ports of MRST ``modules/ad-blackoil/examples``."""

__all__ = [
    "run_blackoil_tutorial_one_phase",
    "run_blackoil_tutorial_grav_seg",
    "run_immiscible_time_integration_example",
    "run_multisegment_well_example",
    "run_simulator_workflow_example",
]


def __getattr__(name):
    if name == "run_blackoil_tutorial_one_phase":
        from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_one_phase import (
            run_blackoil_tutorial_one_phase,
        )
        return run_blackoil_tutorial_one_phase
    if name == "run_blackoil_tutorial_grav_seg":
        from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_grav_seg import (
            run_blackoil_tutorial_grav_seg,
        )
        return run_blackoil_tutorial_grav_seg
    if name == "run_immiscible_time_integration_example":
        from pymrst.modules.ad_blackoil.examples.immiscible_time_integration_example import (
            run_immiscible_time_integration_example,
        )
        return run_immiscible_time_integration_example
    if name == "run_multisegment_well_example":
        from pymrst.modules.ad_blackoil.examples.multisegment_well_example import (
            run_multisegment_well_example,
        )
        return run_multisegment_well_example
    if name == "run_simulator_workflow_example":
        from pymrst.modules.ad_blackoil.examples.simulator_workflow_example import (
            run_simulator_workflow_example_ad as run_simulator_workflow_example,
        )
        return run_simulator_workflow_example
    raise AttributeError(name)
