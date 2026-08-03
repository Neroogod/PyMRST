from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_caseb4():
    return dataset_info_struct(name="CaseB4", filesize=0.5, hasGrid=True,
        hasRock=False, hasFluid=False,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/caseB4.zip",
        description="Case B4 corner-point, stair-step and pillar grid variants.",
        modelType="Corner-point, stair-step and pillar")
