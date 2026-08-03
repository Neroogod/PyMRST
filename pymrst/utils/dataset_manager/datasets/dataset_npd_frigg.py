from math import nan
from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_frigg():
    return dataset_info_struct(name="NPD_Frigg", cells=nan, filesize=42.7,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/nordsjoen/frigg.zip",
        description="North Sea Frigg CO2 storage model.", modelType="Eclipse deck (CO2STORE).")
