"""Resolve examples declared by a dataset definition."""

from dataclasses import dataclass
from pathlib import Path

from pymrst.utils.dataset_manager.utils.get_dataset_info import get_dataset_info


@dataclass(frozen=True)
class DatasetExample:
    name: str
    module: str
    path: str = ""


def list_dataset_examples(name):
    info = name if hasattr(name, "examples") else get_dataset_info(name)[0]
    root = Path(__file__).resolve().parents[4]
    output = []
    for declared in info.examples:
        module, separator, example = declared.partition(":")
        if not separator:
            module, example = "", module
        candidates = list(root.rglob(f"{example}.py"))
        output.append(DatasetExample(example, module,
                                     str(candidates[0]) if candidates else ""))
    return output


listDatasetExamples = list_dataset_examples
