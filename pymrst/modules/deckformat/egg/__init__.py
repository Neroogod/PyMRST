"""Locating and installing the official EGG benchmark dataset.

The deck itself is read by
`pymrst.modules.test_suite.dataset_functions.egg.get_deck_egg`, which is
`readEclipseDeck` + `convertDeckUnits` as `getDeckEGG.m` calls them.
"""

from pymrst.modules.deckformat.egg.egg_dataset import (
    egg_data_directory, fetch_egg_dataset,
)

__all__ = ["egg_data_directory", "fetch_egg_dataset"]
