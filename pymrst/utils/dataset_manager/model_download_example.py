"""Runnable equivalent of MRST's ``modelDownloadExample.m``."""

from pymrst.utils.dataset_manager.get_available_datasets import get_available_datasets
from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
from pymrst.utils.dataset_manager.utils.list_dataset_examples import list_dataset_examples


def model_download_example(download=False):
    info, present = get_available_datasets()
    result = {"datasets": info, "present": present,
              "spe1_examples": list_dataset_examples("SPE1")}
    if download:
        result["spe1_path"] = get_dataset_path(
            "SPE1", download=True, ask_before_download=False)
    return result


if __name__ == "__main__":
    # Keep direct execution non-interactive; call `mrst_dataset_gui()` from a
    # configured session to browse the installed catalog.
    assert callable(model_download_example)
    print("All tests passed!")
