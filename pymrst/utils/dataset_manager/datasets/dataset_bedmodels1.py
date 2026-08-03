from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_bedmodels1():
    return dataset_info_struct(name="BedModels1", filesize=3.3, hasGrid=True,
        hasRock=True, hasFluid=False,
        website="http://brage.bibsys.no/xmlui/handle/11250/259148",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/models_laods.zip",
        description="Benchmark models for local upscaling from Lars Vingli Odsæter's 2013 thesis.",
        modelType="Corner point")
