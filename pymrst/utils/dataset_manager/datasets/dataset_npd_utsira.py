from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_utsira():
    return dataset_info_struct(name="NPD_Utsira", cells=111162, filesize=161,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/nordsjoen/utsira.zip",
        description="North Sea Utsira CO2 storage model.", modelType="Eclipse deck (CO2STORE).")
