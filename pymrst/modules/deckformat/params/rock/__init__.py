"""Rock/deck parameter processing helpers."""

from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
from pymrst.modules.deckformat.params.rock.init_eclipse_rock import (
    EclipseRock,
    init_eclipse_rock,
)
from pymrst.modules.deckformat.params.rock.process_aquifer import (
    AquiferDeckOutput,
    AquiferIndices,
    AquiferProps,
    process_aquifer,
)

__all__ = [
    "AquiferDeckOutput",
    "AquiferIndices",
    "AquiferProps",
    "EclipseRock",
    "compress_rock",
    "init_eclipse_rock",
    "process_aquifer",
]

