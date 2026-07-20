"""MRST `modules/compositional/models` compatibility layout."""

from pymrst.modules.compositional.model import (
    CompositionalState,
    OverallCompositionModel,
)
from pymrst.modules.compositional.natural_variables import (
    GenericNaturalVariablesModel,
    NaturalVariablesCompositionalModel,
    NaturalVariablesState,
)
from pymrst.modules.compositional.models.three_phase_compositional_model import (
    ThreePhaseCompositionalModel,
)
from pymrst.modules.compositional.models.generic_overall_composition_model import (
    GenericOverallCompositionModel,
)

__all__ = [
    "CompositionalState",
    "GenericNaturalVariablesModel",
    "GenericOverallCompositionModel",
    "NaturalVariablesCompositionalModel",
    "NaturalVariablesState",
    "OverallCompositionModel",
    "ThreePhaseCompositionalModel",
    "MODELS", "MRST_CONTENTS_MODELS", "list_models", "list_mrst_contents_models",
]

from pymrst.modules.compositional.models.contents import (
    MODELS,
    MRST_CONTENTS_MODELS,
    list_models,
    list_mrst_contents_models,
)
