from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_sftspe9():
    return dataset_info_struct(name="SFTSPE9", cells=9000, filesize=0.070,
        hasGrid=True, hasRock=True, hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/sftspe9.zip",
        description="Ninth SPE benchmark augmented with surfactant.",
        modelType="Three-phase black-oil with surfactant. Cartesian grid")
