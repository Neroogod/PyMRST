from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_spe3():
    return dataset_info_struct(
        name="SPE3", cells=324, filesize=0.008, hasGrid=True, hasRock=True,
        hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/SPE3.zip",
        examples=("ad-blackoil:simulateSPE3",),
        description="Modified wet-gas black-oil version of the third SPE condensate benchmark.",
        modelType="Three-phase, black-oil with oil component in gas phase. Cartesian grid")

