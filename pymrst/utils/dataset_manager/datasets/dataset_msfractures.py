from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_msfractures():
    return dataset_info_struct(name="MSFractures", cells=7932, filesize=1.549,
        hasGrid=True, hasRock=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/MSFractures.zip",
        description="Grid and permeability for the Moyner–Tchelepi compositional multiscale example.",
        modelType="Grid and permeability in MRST MAT format")
