"""AD-OO simulator framework core (MRST modules/ad-core).

The state-function graph models are assembled from - `StateFunction`,
`StateFunctionGrouping`, the PVT / flow / flux evaluator sets, the
`PhysicalModel` protocol, the facility (wells), and a `NonLinearSolver`
with relaxation, line search and timestep cutting - plus the discrete
operators and the original schedule driver.

Concrete models live in pymrst.modules.ad_blackoil; fluids in
pymrst.modules.ad_props.
"""

from pymrst.modules.ad_core.statefunctions.components.components import (
    Component,
    ConcentrationComponent,
    GasComponent,
    ImmiscibleComponent,
    OilComponent,
    blackoil_components,
)
from pymrst.modules.ad_core.utils.boundary_conditions import (
    ForceContributions,
    add_bc_and_sources,
    boundary_reservoir_fluxes,
    compute_sources_and_boundary_conditions_ad,
    computeSourcesAndBoundaryConditionsAD,
    source_reservoir_fluxes,
)
from pymrst.modules.ad_core.utils.set_weno_discretization import (
    set_weno_discretization,
    setWENODiscretization,
)
from pymrst.modules.ad_core.backends import (
    AutoDiffBackend,
    DiagonalAutoDiffBackend,
    DiagonalJacobian,
    GenericAD,
    RowMajorDiagonalJacobian,
    SparseAutoDiffBackend,
    combine_diagonal_equations,
    init_variables_diagonal,
)
from pymrst.modules.ad_core.initialization import (
    EquilibriumRegion,
    init_eclipse_problem_ad,
    initEclipseProblemAD,
    init_state_equil,
    initialize_equilibrium_pressures,
    initialize_equilibrium_saturations,
)
from pymrst.modules.ad_core.solvers.linear_solvers import (
    CPR_CROSSOVER_BLOCK_JACOBI,
    CPR_CROSSOVER_ILU,
    CPRSolver,
    DirectSolver,
    select_linear_solver,
)
from pymrst.modules.ad_core.solvers.get_non_linear_solver import (
    get_non_linear_solver,
    getNonLinearSolver,
)
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.facilities.separators import (
    BasicSeparator,
    EOSSeparator,
    SeparatorFlashResult,
    SeparatorGroup,
    SeparatorMassResult,
    SeparatorStreamResult,
    SeparatorSurfaceResult,
)
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.simulators.nonlinear_solver import (
    NonLinearSolver,
    SimulationReport,
    StepReport,
)
from pymrst.modules.ad_core.utils.operators import Operators, setup_operators_tpfa
from pymrst.modules.ad_core.timesteps import (
    FactorTimeStepSelector,
    IterationCountTimeStepSelector,
    SimpleTimeStepSelector,
    StateChangeTimeStepSelector,
    rampup_timesteps,
)
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    FlowPropertyFunctions,
)
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    PhaseFluid,
    PVTPropertyFunctions,
    blackoil_phase_fluid,
    simple_phase_fluid,
)
from pymrst.modules.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from pymrst.modules.ad_core.statefunctions.state_function import (
    Dependency,
    State,
    StateFunction,
    StateFunctionGrouping,
)

__all__ = [
    # operators and the original schedule driver
    "Operators", "setup_operators_tpfa", "simulate_schedule_ad",
    "AutoDiffBackend", "DiagonalAutoDiffBackend", "DiagonalJacobian",
    "GenericAD", "RowMajorDiagonalJacobian", "SparseAutoDiffBackend", "combine_diagonal_equations",
    "init_variables_diagonal",
    # the state-function graph
    "StateFunction", "StateFunctionGrouping", "State", "Dependency",
    "PVTPropertyFunctions", "FlowPropertyFunctions", "FlowDiscretization",
    "PhaseFluid", "simple_phase_fluid", "blackoil_phase_fluid",
    # models and solver
    "PhysicalModel", "ReservoirModel", "FacilityModel",
    "BasicSeparator", "EOSSeparator", "SeparatorFlashResult", "SeparatorMassResult",
    "SeparatorGroup", "SeparatorStreamResult", "SeparatorSurfaceResult",
    "ForceContributions", "add_bc_and_sources", "boundary_reservoir_fluxes",
    "compute_sources_and_boundary_conditions_ad",
    "computeSourcesAndBoundaryConditionsAD", "source_reservoir_fluxes",
    "set_weno_discretization", "setWENODiscretization",
    "EquilibriumRegion", "init_state_equil",
    "init_eclipse_problem_ad", "initEclipseProblemAD",
    "initialize_equilibrium_pressures", "initialize_equilibrium_saturations",
    # components
    "Component", "ImmiscibleComponent", "GasComponent", "OilComponent",
    "ConcentrationComponent", "blackoil_components",
    # linear solvers
    "CPRSolver", "DirectSolver", "select_linear_solver",
    "get_non_linear_solver", "getNonLinearSolver",
    "CPR_CROSSOVER_BLOCK_JACOBI", "CPR_CROSSOVER_ILU",
    # adaptive timestepping
    "SimpleTimeStepSelector", "FactorTimeStepSelector",
    "IterationCountTimeStepSelector", "StateChangeTimeStepSelector",
    "rampup_timesteps",
    "NonLinearSolver", "StepReport", "SimulationReport",
]
