from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_igemsgrids():
    return dataset_info_struct(name="IGEMS grids", cells=3600000, filesize=470,
        hasGrid=True, hasRock=True, website="https://www.nr.no/IGEMS/",
        fileurl="http://files.nr.no/igems/eclipsegrids.zip",
        description="Full IGEMS suite of stochastic geological corner-point grids.", modelType="Corner-point")
