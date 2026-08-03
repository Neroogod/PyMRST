from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_sleipnerplumes():
    return dataset_info_struct(name="SleipnerPlumes", filesize=1,
        website="http://www.ieaghg.org/networks/modelling-network",
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/sleipner_plume_thicknesses.zip",
        description="Processed plume-thickness observations; this is plume data rather than the IEAGHG grid model.",
        modelType="Plume data only. No grid model")
