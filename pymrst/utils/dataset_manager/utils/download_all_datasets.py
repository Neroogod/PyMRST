"""Install every directly downloadable missing dataset."""

from pymrst.utils.dataset_manager.download_dataset import download_dataset
from pymrst.utils.dataset_manager.get_available_datasets import get_available_datasets
from pymrst.utils.dataset_manager.utils.predicates import dataset_has_custom_download_fcn, dataset_has_valid_file_url


def download_all_datasets(ask_first=True):
    info, present = get_available_datasets()
    pending = [item for item, installed in zip(info, present)
               if not installed and (dataset_has_valid_file_url(item)
                                     or dataset_has_custom_download_fcn(item))]
    if ask_first and pending:
        size = sum(item.filesize for item in pending
                   if item.filesize is not None and item.filesize == item.filesize)
        answer = input(f"Download all {len(pending)} datasets ({size:.0f} MB)? [Y/n] ")
        if answer.strip().lower() not in ("", "y", "yes"):
            return {}
    return {item.name: download_dataset(item.name, ask_first=False)[1]
            for item in pending}


downloadAllDatasets = download_all_datasets
