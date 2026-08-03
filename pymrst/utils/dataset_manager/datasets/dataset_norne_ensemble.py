from pymrst.utils.dataset_manager.utils import dataset_info_struct
def dataset_norne_ensemble():
    return dataset_info_struct(name="Norne_Ensemble", filesize=5.6,
        website="https://github.com/rolfjl/Norne-Initial-Ensemble",
        instructions="Install the Norne initial ensemble from its GitHub repository.",
        description="Initial geological ensemble for uncertainty studies of the Norne field.",
        modelType="Three-phase, black-oil, corner-point")
