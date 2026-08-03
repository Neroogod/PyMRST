"""Standard dataset metadata (MRST ``datasetInfoStruct.m``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from pymrst.utils.settings_manager import mrst_settings


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str = ""
    cells: Optional[int] = None
    website: str = ""
    fileurl: str = ""
    filesize: Optional[float] = None
    instructions: str = ""
    hasGrid: bool = False
    hasRock: bool = False
    hasFluid: bool = False
    source: str = ""
    image: str = ""
    examples: tuple[str, ...] = field(default_factory=tuple)
    modelType: str = "Unknown"
    downloadFcn: Optional[Callable[[], bool]] = None
    note: tuple[str, ...] = field(default_factory=tuple)


def dataset_info_struct(**values):
    values["name"] = "".join(str(values.get("name", "")).split())
    for name in ("examples", "note"):
        if name in values and not isinstance(values[name], tuple):
            values[name] = tuple(values[name]) if values[name] else ()
    info = DatasetInfo(**values)
    path = Path(mrst_settings("get", "dataDirectory")) / info.name
    return info, path.is_dir()


datasetInfoStruct = dataset_info_struct

