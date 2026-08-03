"""Resolve or install a registered dataset (MRST ``getDatasetPath.m``)."""

from pathlib import Path

from pymrst.utils.settings_manager import mrst_settings

from pymrst.utils.dataset_manager.download_dataset import download_dataset
from pymrst.utils.dataset_manager.utils import get_dataset_info


def get_dataset_path(name, download=True, ask_before_download=None,
                     skip_available_check=False):
    info, present = get_dataset_info(name)
    path = Path(mrst_settings("get", "dataDirectory")) / info.name
    if present or skip_available_check:
        return path
    if download:
        ask = (mrst_settings("get", "promptDL") if ask_before_download is None
               else ask_before_download)
        path, ok = download_dataset(name, ask)
        if ok:
            return path
    raise FileNotFoundError(
        f"dataset {info.name!r} is not installed at {path}; "
        f"download it with download_dataset({info.name!r})")


getDatasetPath = get_dataset_path

