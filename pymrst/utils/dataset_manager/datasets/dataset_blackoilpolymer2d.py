from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_blackoilpolymer2d():
    return dataset_info_struct(name="BlackoilPolymer2d", cells=100,
        filesize=1.7, hasGrid=True, hasRock=True, hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/blackoilpolymer2d.zip",
        description="Two-dimensional black-oil polymer injection cases with and without shear effects and ECLIPSE references.",
        modelType="Three-phase, black-oil with polymer injection. Cartesian grid")
