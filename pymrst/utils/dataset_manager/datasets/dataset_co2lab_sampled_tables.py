from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_co2lab_sampled_tables():
    return dataset_info_struct(name="CO2lab_Sampled_Tables", filesize=102.8,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/co2lab_sampled_tables.zip",
        description="Precomputed sampled property tables for the CO2lab module.",
        modelType="MAT files with tables")
