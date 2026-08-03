from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_spe10_layer_compositional():
    return dataset_info_struct(name="SPE10_Layer_Compositional", filesize=7.6,
        hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/SPE10_Layer_Compositional.zip",
        description="Water-alternating-gas compositional case for layer 1 of SPE10 Model 2.")
