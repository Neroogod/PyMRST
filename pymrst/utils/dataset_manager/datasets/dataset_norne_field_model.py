from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_norne_field_model():
    return dataset_info_struct(
        name="norne_field_model", filesize=69.0, hasGrid=True, hasRock=True,
        hasFluid=True,
        fileurl="https://www.sintef.no/globalassets/project/mrst/norne_field_model_example_files.zip",
        examples=("ad-blackoil:fieldModelNorneExample",),
        description="Data files required by the fieldModelNorne example.",
        modelType="Three-phase, black-oil, corner-point")
