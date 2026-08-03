"""Dataset discovery and installation following MRST's layout."""

from pymrst.utils.dataset_manager.download_dataset import download_dataset, downloadDataset
from pymrst.utils.dataset_manager.get_available_datasets import get_available_datasets, getAvailableDatasets
from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path, getDatasetPath
from pymrst.utils.dataset_manager.utils import DatasetInfo, get_dataset_info

__all__ = ["DatasetInfo", "download_dataset", "downloadDataset",
           "get_available_datasets", "getAvailableDatasets",
           "get_dataset_info", "get_dataset_path", "getDatasetPath"]

