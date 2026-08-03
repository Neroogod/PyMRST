from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_stordbasin():
    return dataset_info_struct(name="NPD_Stordbasin", cells=106365, filesize=113,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/nordsjoen/stordbasin.zip",
        description="North Sea Stord Basin CO2 storage model.", modelType="Eclipse deck (CO2STORE).")
