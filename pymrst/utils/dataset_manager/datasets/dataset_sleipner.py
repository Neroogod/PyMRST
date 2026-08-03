from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_sleipner():
    return dataset_info_struct(name="Sleipner", cells=262990, hasGrid=True,
        hasRock=True, website="http://www.ieaghg.org/networks/modelling-network",
        instructions="Obtain the IEAGHG Sleipner benchmark and install it in the Sleipner dataset directory.",
        description="IEAGHG benchmark model of CO2 injection into the Utsira Formation at Sleipner.",
        modelType="Two-phase CO2-water, corner-point")
