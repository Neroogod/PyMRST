"""SPE10 benchmark dataset access."""

from pymrst.modules.deckformat.spe10.model1 import (
    SPE10_URL, fetch_spe10_dataset, load_spe10_model1,
    spe10_data_directory,
)

__all__ = [
    "SPE10_URL", "fetch_spe10_dataset", "load_spe10_model1",
    "spe10_data_directory",
]
