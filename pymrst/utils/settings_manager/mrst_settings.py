"""Configure persistent settings (MRST ``settings_manager/mrstSettings.m``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterator

from pymrst.utils.settings_manager.settings_struct import SettingsStruct

_LOCK = RLock()
_SETTINGS = None


def _settings_file() -> Path:
    root = Path(os.environ.get("PYMRST_CONFIG_DIR", Path.home() / ".pymrst"))
    return root.expanduser() / "settings.json"


def _load() -> SettingsStruct:
    path = _settings_file()
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        known = SettingsStruct.__dataclass_fields__
        return SettingsStruct(**{k: v for k, v in values.items() if k in known}).validate()
    return SettingsStruct(dataDirectory=str(path.parent / "datasets"),
                          outputDirectory=str(path.parent / "output")).validate()


def _store(settings: SettingsStruct) -> None:
    path = _settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings.to_dict(), indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def mrst_settings(verb="get", name=None, value=None):
    """Get, set, list, or reset persistent PyMRST settings."""
    global _SETTINGS
    with _LOCK:
        if _SETTINGS is None:
            _SETTINGS = _load()
        command = (verb or "get").lower()
        if command in ("setup", "reset"):
            root = _settings_file().parent
            _SETTINGS = SettingsStruct(
                dataDirectory=str(root / "datasets"),
                outputDirectory=str(root / "output")).validate()
            _store(_SETTINGS)
            return _SETTINGS.to_dict()
        if command in ("list", ""):
            return _SETTINGS.to_dict()
        if command == "get":
            if name is None:
                return _SETTINGS.to_dict()
            if not hasattr(_SETTINGS, name):
                raise KeyError(f"unknown PyMRST setting {name!r}")
            return getattr(_SETTINGS, name)
        if command == "set":
            if not hasattr(_SETTINGS, name):
                raise KeyError(f"unknown PyMRST setting {name!r}")
            current = getattr(_SETTINGS, name)
            if isinstance(current, bool) and isinstance(value, str):
                lowered = value.lower()
                value = not current if lowered == "toggle" else lowered in ("true", "on")
                if lowered not in ("toggle", "true", "on", "false", "off"):
                    raise ValueError(f"invalid logical setting value {value!r}")
            setattr(_SETTINGS, name, value)
            _SETTINGS.validate()
            _store(_SETTINGS)
            return None
        raise ValueError(f"unknown settings verb {verb!r}")


mrstSettings = mrst_settings


@contextmanager
def override_setting(name: str, value: Any) -> Iterator[None]:
    """Use `value` for setting `name` in this process only, restoring it on exit.

    Not an MRST function. `mrst_settings("set", ...)` persists to the settings
    file, so a test that set and later restored a value could leave another,
    concurrently running process's temporary value behind in the user's file.
    This changes the in-memory settings only.
    """
    global _SETTINGS
    with _LOCK:
        if _SETTINGS is None:
            _SETTINGS = _load()
        if not hasattr(_SETTINGS, name):
            raise KeyError(f"unknown PyMRST setting {name!r}")
        previous = getattr(_SETTINGS, name)
        setattr(_SETTINGS, name, value)
    try:
        yield
    finally:
        with _LOCK:
            setattr(_SETTINGS, name, previous)
