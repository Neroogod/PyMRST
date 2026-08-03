from math import nan
from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_co2labmit():
    return dataset_info_struct(name="CO2LabMIT", filesize=nan, hasGrid=True,
        hasRock=False, hasFluid=False,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/CO2LabMIT.zip",
        description="Input datasets for examples in the co2lab-mit module.",
        modelType="Input files and tables")
