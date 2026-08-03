from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_olympus():
    return dataset_info_struct(name="Olympus", cells=192750, hasGrid=True,
        hasRock=True, hasFluid=True,
        website="http://www.isapp2.com/optimization-challenge.html",
        instructions="Download the Olympus field optimization challenge and extract it into the Olympus dataset directory.",
        description="Fifty geological realizations for the Olympus field optimization challenge.",
        modelType="Two-phase oil-water, corner-point")
