from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_npd_egersund():
    return dataset_info_struct(name="NPD_Egersund", cells=105310, filesize=157,
        hasGrid=True, hasRock=True, hasFluid=True,
        website="https://www.sodir.no/fakta/co2-handtering-ccs/lagringsmodeller/",
        fileurl="https://www.sodir.no/globalassets/1-sodir/fakta/co-to/lagringsmodeller/nordsjoen/egersund.zip",
        description="North Sea Egersund CO2 storage model.", modelType="Eclipse deck (CO2STORE).")
