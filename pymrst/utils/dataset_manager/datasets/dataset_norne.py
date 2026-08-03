from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_norne():
    return dataset_info_struct(
        name="Norne", cells=44420, filesize=20.0, hasGrid=True, hasRock=True,
        hasFluid=True,
        website="http://www.ipt.ntnu.no/~norne/wiki/doku.php?id=english:nornebenchmarkcase2",
        examples=("showNorne", "book:runNorneSynthetic", "incomp:incompExampleNorne1ph", "incomp:incompExampleNorne2ph"),
        description="Open simulation model of the Statoil-operated Norne field in the Norwegian Sea.",
        instructions="Install the OPM Norne deck in the Norne dataset directory.",
        modelType="Three-phase, black-oil, corner-point")

