from pathlib import Path

from pymrst.utils.settings_manager import mrst_settings

from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_egg():
    root = Path(mrst_settings("get", "dataDirectory"))
    return dataset_info_struct(
        name="Egg", cells=18553, hasGrid=True, hasRock=True, hasFluid=True,
        website="https://data.4tu.nl/articles/dataset/The_Egg_Model_-_data_files/12707642",
        instructions=f'Download Egg_Model_Data_Files_v2 and extract it as "Egg" under {root}.',
        modelType="Two-phase oil-water, corner-point")

