from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_simplecomp():
    return dataset_info_struct(name="simplecomp", cells=1000, filesize=38.2,
        hasGrid=True, hasRock=True, hasFluid=True, website="www.sintef.no/mrst",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/simple_comp.zip",
        description="Simple one-dimensional compositional CO2/methane/decane problem.",
        modelType="Two-phase compositional model. Cartesian 1D grid")
