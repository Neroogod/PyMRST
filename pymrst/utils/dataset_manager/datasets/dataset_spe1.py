from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_spe1():
    return dataset_info_struct(
        name="SPE1", cells=300, filesize=0.005, hasGrid=True, hasRock=True,
        hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/SPE1.zip",
        examples=("ad-blackoil:runAndCompareSPE1", "ad-blackoil:demonstrateExtraOutputSPE1", "ad-blackoil:timestepComparisonSPE1"),
        description="First SPE benchmark: live-oil/dry-gas black-oil model with nearly immobile water.",
        modelType="Three-phase, black-oil with gas in oil phase. Cartesian grid")

