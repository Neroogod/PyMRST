from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_aquifertest():
    return dataset_info_struct(name="aquifertest", hasGrid=True, hasRock=True,
        hasFluid=True, fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/aquifertestdata.zip",
        description="Simple reservoir model with an aquifer for testing aquifer functionality.",
        modelType="Two-phase with aquifer. Cartesian grid")
