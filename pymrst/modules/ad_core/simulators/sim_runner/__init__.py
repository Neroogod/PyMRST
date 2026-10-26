"""MRST ``ad-core/simulators/sim_runner``: disk-backed packed simulation problems.

The MRST-named functions are the ports for the AD-OO models
(`initEclipsePackedProblemAD`, `packSimulationProblem`,
`simulatePackedProblem`, `getPackedSimulatorOutput`,
`getMultiplePackedSimulatorOutputs`, `plotPackedProblem`,
`clearPackedSimulatorOutput`, `copyPackedProblem`,
`simulatePackedProblemStandalone`, `simulatePackedProblemBackground`,
`monitorBackgroundSimulations`). The `*_ad`
functions are PyMRST's earlier versions for the legacy models.
"""

from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import (
    PackedSimulationProblem,
    ResultHandler,
    pack_simulation_problem_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
    packSimulationProblem,
    pack_simulation_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_output import (
    clear_packed_simulator_output_ad,
    get_packed_simulator_output_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
    clear_packed_simulator_output,
    clearPackedSimulatorOutput,
)
from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
    getPackedSimulatorOutput,
    get_packed_simulator_output,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    PackedSimulationStatus,
    simulate_packed_problem,
    simulate_packed_problem_ad,
    simulatePackedProblem,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_collection import (
    copy_packed_problem_ad,
    get_multiple_packed_simulator_outputs_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.copy_packed_problem import (
    copy_packed_problem,
    copyPackedProblem,
)
from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
    getMultiplePackedSimulatorOutputs,
    get_multiple_packed_simulator_outputs,
)
from pymrst.modules.ad_core.simulators.sim_runner.plot_packed_problem import (
    plotPackedProblem,
    plot_packed_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.background import (
    BackgroundSimulationInfo,
    monitor_background_simulations_ad,
    simulate_packed_problem_background_ad,
    simulate_packed_problem_standalone_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.monitor_background_simulations import (
    monitor_background_simulations,
    monitorBackgroundSimulations,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem_background import (
    simulate_packed_problem_background,
    simulatePackedProblemBackground,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem_standalone import (
    simulate_packed_problem_standalone,
    simulatePackedProblemStandalone,
)
from pymrst.modules.ad_core.simulators.sim_runner.init_eclipse_packed_problem_ad import (
    init_eclipse_packed_problem_ad,
    init_eclipse_packed_problem_ad_ad,
    initEclipsePackedProblemAD,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager import (
    PackedProblemManager,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad import (
    PackedProblemManagerAD,
)

__all__ = [
    "PackedSimulationProblem", "ResultHandler",
    "pack_simulation_problem", "pack_simulation_problem_ad", "packSimulationProblem",
    "clear_packed_simulator_output", "clear_packed_simulator_output_ad",
    "clearPackedSimulatorOutput",
    "get_packed_simulator_output", "get_packed_simulator_output_ad",
    "getPackedSimulatorOutput",
    "PackedSimulationStatus", "simulate_packed_problem", "simulate_packed_problem_ad",
    "simulatePackedProblem",
    "copy_packed_problem", "copy_packed_problem_ad", "copyPackedProblem",
    "get_multiple_packed_simulator_outputs", "get_multiple_packed_simulator_outputs_ad",
    "getMultiplePackedSimulatorOutputs",
    "plot_packed_problem", "plotPackedProblem",
    "BackgroundSimulationInfo", "monitor_background_simulations",
    "monitorBackgroundSimulations", "simulate_packed_problem_background",
    "simulatePackedProblemBackground", "simulate_packed_problem_standalone",
    "simulatePackedProblemStandalone", "monitor_background_simulations_ad",
    "simulate_packed_problem_background_ad", "simulate_packed_problem_standalone_ad",
    "init_eclipse_packed_problem_ad", "init_eclipse_packed_problem_ad_ad",
    "initEclipsePackedProblemAD",
    "PackedProblemManager", "PackedProblemManagerAD",
]
