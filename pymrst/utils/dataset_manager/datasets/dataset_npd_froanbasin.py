from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_froanbasin():
    return dataset_info_struct(name="NPD_Froanbasin", cells=350000, filesize=6.7,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/norskehavet/froan.zip",
        description="Norwegian Sea Froan Basin CO2 storage model.", modelType="Eclipse deck (CO2STORE).")
