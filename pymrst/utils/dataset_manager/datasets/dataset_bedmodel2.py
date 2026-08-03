from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_bedmodel2():
    return dataset_info_struct(name="BedModel2", cells=91831, filesize=3.4,
        hasGrid=True, hasRock=True, hasFluid=False,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/bedModel2.zip",
        description="Highly detailed sedimentary bed model for flow and upscaling studies.",
        modelType="Corner-point")
