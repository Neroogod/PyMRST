from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_co2atlasbarentssea():
    return dataset_info_struct(name="CO2Atlas Barents Sea", filesize=4.3,
        hasGrid=True, hasRock=False, hasFluid=False,
        website="http://www.npd.no/en/Publications/Reports/CO2-storage-atlas-Barents-Sea/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/CO2Atlas_BarentsSea.zip",
        description="NPD depth and thickness maps for the Barents Sea CO2 storage atlas.",
        modelType="Depth and thickness maps")
