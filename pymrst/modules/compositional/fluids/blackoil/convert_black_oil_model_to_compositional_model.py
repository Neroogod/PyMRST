"""Port of MRST ``convertBlackOilModelToCompositionalModel.m``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.fluids.blackoil.convert_rs_rv_to_k_value import (
    RSRVKValueProperties,
    convert_rs_rv_to_k_value,
)
from pymrst.modules.compositional.fluids.blackoil.kvalue_model import (
    BlackOilPropertyModel,
    EquilibriumConstantModel,
)


@dataclass
class BlackOilCompositionalModel:
    """Lightweight native K-value compositional model built from a BO model."""

    blackoil_model: object
    mixture: object
    props: RSRVKValueProperties
    property_model: BlackOilPropertyModel
    eos_model: EquilibriumConstantModel

    @property
    def G(self):
        return self.blackoil_model.G

    @property
    def rock(self):
        return self.blackoil_model.rock


def pseudo_blackoil_mixture() -> Mixture:
    """MRST pseudo mixture used by ``convertBlackOilModelToCompositionalModel``."""
    return Mixture(
        names=("PseudoOil", "PseudoGas"),
        Tcrit=np.array([np.nan, np.nan]),
        Pcrit=np.array([np.nan, np.nan]),
        Vcrit=np.array([np.nan, np.nan]),
        acentric=np.array([np.nan, np.nan]),
        molarMass=np.array([1.0, 1.0]),
    )


def convert_black_oil_model_to_compositional_model(
    blackoil_model,
    **kwargs,
) -> BlackOilCompositionalModel:
    """Build PyMRST's native counterpart of MRST's BO-to-compositional K model."""
    props = convert_rs_rv_to_k_value(blackoil_model, **kwargs)
    mixture = pseudo_blackoil_mixture()
    property_model = BlackOilPropertyModel(props)
    eos_model = EquilibriumConstantModel(mixture, props, property_model)
    return BlackOilCompositionalModel(
        blackoil_model=blackoil_model,
        mixture=mixture,
        props=props,
        property_model=property_model,
        eos_model=eos_model,
    )


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial
    from pymrst.modules.compositional.fluids.blackoil import convert_black_oil_state_to_compositional

    def test_convert_spe1_model_to_kvalue_compositional():
        _, model, state = setup_spe1_tutorial()
        comp_model = convert_black_oil_model_to_compositional_model(model, np_=12, nz=16)
        comp_state = convert_black_oil_state_to_compositional(model, state)
        K = comp_model.eos_model.evaluate_equilibrium_constants(
            comp_state.pressure[:5],
            np.full(5, 273.15),
            comp_state.z[:5],
        )
        assert comp_model.mixture.names == ("PseudoOil", "PseudoGas")
        assert K.shape == (5, 2)
        assert np.all(np.isfinite(K))
        print("ok test_convert_spe1_model_to_kvalue_compositional")

    test_convert_spe1_model_to_kvalue_compositional()
    print("\nAll tests passed!")
