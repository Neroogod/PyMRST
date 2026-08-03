from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_co2atlasnorwegiansea():
    return dataset_info_struct(name="CO2Atlas Norwegian Sea", filesize=36.8,
        hasGrid=True, hasRock=False, hasFluid=False,
        website="http://www.npd.no/en/Publications/Reports/CO2-storage-altas-Norwegian-Sea/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/CO2Atlas_NorwegianSea.zip",
        description="NPD depth and thickness maps for the Norwegian Sea CO2 storage atlas.",
        modelType="Depth and thickness maps")
