"""Discover registered datasets (MRST ``getAvailableDatasets.m``)."""

from importlib import import_module
from pkgutil import iter_modules


def get_available_datasets():
    # Imported here rather than at module scope. The package `__init__` imports
    # this module, so reaching back up to `pymrst.utils.dataset_manager` for the
    # `datasets` subpackage closes an import cycle: it works, because Python
    # serves the half-built parent, but it is the sixth cycle in the codebase
    # and the only one that is this easy to remove.
    from pymrst.utils.dataset_manager import datasets

    found = []
    for module in sorted(iter_modules(datasets.__path__), key=lambda item: item.name):
        if not module.name.startswith("dataset_"):
            continue
        loaded = import_module(f"{datasets.__name__}.{module.name}")
        found.append(getattr(loaded, module.name)())
    return [item[0] for item in found], [item[1] for item in found]


getAvailableDatasets = get_available_datasets

