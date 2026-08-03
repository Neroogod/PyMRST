from math import nan
from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_bjarmeland():
    return dataset_info_struct(name="NPD_Bjarmeland", cells=nan, filesize=370,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/barentshavet/bjarmeland.zip",
        description="Barents Sea CO2 storage model from the Norwegian Offshore Directorate.", modelType="Eclipse deck (CO2STORE).")
