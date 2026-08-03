"""Persistent PyMRST settings (MRST ``settings_manager/settingsStruct.m``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SettingsStruct:
    allowDL: bool = True
    dataDirectory: str = ""
    outputDirectory: str = ""
    promptDL: bool = True
    promptMEX: bool = True
    useMEX: bool = False
    useOMP: bool = True
    useHash: bool = True

    def __post_init__(self):
        base = Path.home() / ".pymrst"
        if not self.dataDirectory:
            self.dataDirectory = str(base / "datasets")
        if not self.outputDirectory:
            self.outputDirectory = str(base / "output")

    def validate(self) -> "SettingsStruct":
        for name in ("dataDirectory", "outputDirectory"):
            path = Path(getattr(self, name)).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, name, str(path.resolve()))
        return self

    def to_dict(self):
        return asdict(self)


settingsStruct = SettingsStruct

