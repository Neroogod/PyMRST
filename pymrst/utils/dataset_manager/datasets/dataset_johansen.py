from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_johansen():
    return dataset_info_struct(name="Johansen", filesize=39.2, hasGrid=True,
        hasRock=True, website="https://www.sintef.no/Projectweb/MatMorA/Downloads/Johansen/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/johansen.zip",
        description="Whole-field and sector models of the Johansen offshore CO2-storage formation.", modelType="Corner-point")
