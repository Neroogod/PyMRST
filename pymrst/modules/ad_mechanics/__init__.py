"""Coupled flow and geomechanics (MRST modules/ad-mechanics).

- `MechanicModel`: poroelastic mechanics on the VEM discretization -
  displacement in equilibrium with a given pressure field.
- `MechWaterModel`: fully coupled single-phase poroelasticity. In the
  state-function graph the entire coupling is one property,
  `PoroMechanicalPoreVolume`: the pore volume grows with the rock's
  dilation. Nothing downstream knows that mechanics happened.
- `MechBlackOilModel`: the multiphase version (MRST's `MechOilWaterModel`
  and `MechBlackOilModel`) - the validated `GenericBlackOilModel` with the
  same pore volume installed and the displacement added as a primary
  variable. **Note:** MRST's `equationsOilWaterMech` omits the cell volume
  on its Biot term; PyMRST is correct, and `mrst_oilwater_bug=True`
  reproduces MRST's number.
- `FixedStressSplitModel`: the sequential scheme, solving mechanics and
  flow in turn with the fixed-stress compressibility kept implicit. It
  converges to the fully coupled answer, which is what makes it usable at
  field scale.
- `MandelCase`: the benchmark that discriminates - a wrongly coupled model
  passes Terzaghi and fails Mandel.
- `compute_gradient_adjoint` / `uplift_objective`: the gradient of surface
  uplift with respect to every well control, from one backward sweep -
  which is how coupled geomechanics is actually put to work (how should the
  wells run so the ground stays still?).
"""

from pymrst.modules.ad_mechanics.adjoint_impl import (
    AdjointResult,
    compute_gradient_adjoint,
    evaluate_objective,
    simulate_forward,
    uplift_objective,
)
from pymrst.modules.ad_mechanics.mandel import (
    MandelCase,
    consolidation_coefficient,
    mandel_pressure,
    mandel_roots,
    poro_params,
)
from pymrst.modules.ad_mechanics.linear_solver import PoroelasticSchurSolver
from pymrst.modules.ad_mechanics.elastoplastic import (
    ElastoplasticResult,
    ElastoplasticStep,
    solve_elastoplastic_vem,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import (
    MechBlackOilModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import (
    MechOilWaterModel,
)
from pymrst.modules.ad_mechanics.models.mechanic_model import (
    MechanicModel,
    MechanicOperators,
)
from pymrst.modules.mpsaw import (
    MPSAAssembly,
    MPSAInteractionRegion,
    MPSAWResult,
    mpsaw_assemble,
    mpsaw_cell_gradient,
    mpsaw_divergence_operator,
    mpsaw_face_tractions,
    mpsaw_interaction_regions,
    mpsaw_isotropic_stress,
    mpsaw_recover,
    mpsaw_residual,
    mpsaw_strain,
)
from pymrst.modules.ad_mechanics.plasticity import (
    PlasticityState,
    empty_plasticity_state,
    isotropic_elastic_stress,
    von_mises_equivalent,
    von_mises_return_map,
)
from pymrst.modules.ad_mechanics.models.poromechanics import (
    FixedStressPoreVolume,
    FixedStressSplitModel,
    PoroMechanicalPoreVolume,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_water_model import (
    MechWaterModel,
)

__all__ = [
    "MechanicModel", "MechanicOperators",
    "MechWaterModel", "MechBlackOilModel", "MechOilWaterModel", "FixedStressSplitModel",
    "PoroelasticSchurSolver",
    "PoroMechanicalPoreVolume", "FixedStressPoreVolume",
    "MandelCase", "mandel_pressure", "mandel_roots", "poro_params",
    "consolidation_coefficient",
    "compute_gradient_adjoint", "uplift_objective", "evaluate_objective",
    "simulate_forward", "AdjointResult",
    "ElastoplasticResult", "ElastoplasticStep", "solve_elastoplastic_vem",
    "PlasticityState", "empty_plasticity_state", "isotropic_elastic_stress",
    "von_mises_equivalent", "von_mises_return_map",
    "MPSAWResult", "MPSAInteractionRegion", "MPSAAssembly",
    "mpsaw_interaction_regions", "mpsaw_assemble",
    "mpsaw_divergence_operator", "mpsaw_cell_gradient", "mpsaw_strain",
    "mpsaw_isotropic_stress", "mpsaw_face_tractions", "mpsaw_residual",
    "mpsaw_recover",
]
