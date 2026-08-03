from pymrst.utils.dataset_manager.utils import dataset_info_struct


def dataset_eor_book_ii():
    return dataset_info_struct(name="EOR_Book_II", filesize=0.780,
        hasGrid=True, hasRock=True, hasFluid=True,
        fileurl="https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/EOR_Book_II.zip",
        description="Test cases for the ad-eor module.")
