"""Retrieve one registered dataset (MRST ``getDatasetInfo.m``)."""

from importlib import import_module


def get_dataset_info(name):
    key = str(name).lower()
    try:
        module = import_module(f"pymrst.utils.dataset_manager.datasets.dataset_{key}")
        return getattr(module, f"dataset_{key}")()
    except (ModuleNotFoundError, AttributeError) as error:
        raise KeyError(f"dataset {name!r} is not known to PyMRST") from error


getDatasetInfo = get_dataset_info

