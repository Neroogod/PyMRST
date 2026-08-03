from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_co2atlas():
    return dataset_info_struct(name="CO2Atlas", filesize=15.6, hasGrid=True,
        hasRock=False, hasFluid=False,
        website="http://www.npd.no/en/Publications/Reports/CO2-Storage-Atlas-/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/co2atlas.zip",
        description="NPD depth and thickness maps for formations in the Norwegian North Sea CO2 storage atlas.",
        modelType="Depth and thickness maps")
