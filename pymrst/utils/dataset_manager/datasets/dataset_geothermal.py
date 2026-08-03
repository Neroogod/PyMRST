from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_geothermal():
    return dataset_info_struct(name="geothermal", filesize=9.2, hasGrid=True,
        hasRock=True, hasFluid=False,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/geothermal.zip",
        description="Data for geothermal benchmark, EGS and HT-ATES examples.")
