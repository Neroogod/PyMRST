from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_igemssurfaces():
    return dataset_info_struct(name="IGEMS surfaces", cells=180000, filesize=600,
        hasGrid=True, website="https://www.nr.no/IGEMS/",
        fileurl="http://files.nr.no/igems/surfaces.zip",
        description="Geological surfaces from the IGEMS model suite.", modelType="Surface")
