from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_sleipner2019():
    return dataset_info_struct(name="Sleipner2019", cells=1986176,
        hasGrid=True, hasRock=True,
        website="https://co2datashare.org/dataset/sleipner-2019-benchmark-model",
        instructions="Download the Sleipner 2019 benchmark from CO2 DataShare and install it in Sleipner2019.",
        description="Reference geological model and monitoring data for the Sleipner CO2 storage site.",
        modelType="Corner-point")
