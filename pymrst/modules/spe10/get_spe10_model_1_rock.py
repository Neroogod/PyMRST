"""MRST ``spe10/getSPE10_model_1_rock.m``: the rock of the tenth SPE CSP's
Model I (100x1x20 cells, porosity 0.2, a geostatistical permeability), read
from the dataset's `model1_data.mat`.

MRST reads it from `getDatasetPath('spe10')`. PyMRST's data directory may not
hold the dataset while MRST's tree does, so that is looked in too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.rock.rock import Rock


def _model1_data() -> dict:
    from pymrst.modules.deckformat.spe10.model1 import spe10_data_directory
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    path = Path(get_dataset_path("spe10", download=False, skip_available_check=True))
    if not (path/"model1_data.mat").is_file():
        path = spe10_data_directory()
    return loadmat(path/"model1_data.mat", squeeze_me=True, struct_as_record=False)


def get_spe10_model_1_rock() -> Rock:
    """MRST `rock = getSPE10_model_1_rock()`: `perm` (2000x3, SI) and `poro`."""
    data = _model1_data()["rock"]
    return Rock(perm=np.asarray(data.perm, dtype=float),
                poro=np.asarray(data.poro, dtype=float).ravel())


getSPE10_model_1_rock = get_spe10_model_1_rock
__all__ = ["getSPE10_model_1_rock", "get_spe10_model_1_rock"]


if __name__ == "__main__":
    def test_model_one_rock():
        rock = get_spe10_model_1_rock()
        assert rock.perm.shape == (2000, 3) and rock.poro.shape == (2000,)
        assert np.all(rock.poro == 0.2)
        print("ok test_model_one_rock")

    test_model_one_rock()
    print("\nAll tests passed!")
