from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_saigup():
    return dataset_info_struct(name="SAIGUP", cells=78720, filesize=2.535,
        hasGrid=True, hasRock=True,
        website="https://www.sintef.no/projectweb/mrst/downloadable-resources/download-saigup-data-set/",
        fileurl="https://www.sintef.no/globalassets/project/mrst/saigup.tar.gz",
        description="Faulted SAIGUP shallow-marine realization with inactive cells and disconnected components.",
        modelType="Corner-point")
