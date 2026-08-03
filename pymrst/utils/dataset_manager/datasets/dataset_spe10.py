from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_spe10():
    return dataset_info_struct(
        name="SPE10", cells=1122000, filesize=48.1, hasGrid=True, hasRock=True,
        hasFluid=True, website="http://www.spe.org/web/csp/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/SPE10.zip",
        examples=("showSPE10", "spe10:SPE10_rock", "spe10:SPE10_setup", "spe10:simulate_spe10"),
        description="Highly heterogeneous tenth SPE benchmark, originally designed as an upscaling test.",
        modelType="Two-phase fluid, Cartesian grid")

