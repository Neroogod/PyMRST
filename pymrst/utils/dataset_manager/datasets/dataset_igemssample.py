from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_igemssample():
    return dataset_info_struct(name="IGEMSsample", cells=3600000, filesize=9.7,
        hasGrid=True, website="https://www.nr.no/IGEMS/",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/igems_sample.zip",
        description="Sample realization from the IGEMS geological model suite.", modelType="Corner-point")
