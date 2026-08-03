from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_spe9():
    return dataset_info_struct(
        name="SPE9", cells=9000, filesize=0.070, hasGrid=True, hasRock=True,
        hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/SPE9.zip",
        examples=("ad-blackoil:baselineComparisonSPE9",),
        description="Ninth SPE benchmark with one water injector, 25 producers and dynamic well controls.",
        modelType="Three-phase, black-oil with gas in oil phase. Cartesian grid")

