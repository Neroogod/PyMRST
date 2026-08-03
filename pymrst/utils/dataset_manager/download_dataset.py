"""Download and safely extract a registered dataset (MRST ``downloadDataset.m``)."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from pymrst.utils.settings_manager import mrst_settings

from pymrst.utils.dataset_manager.utils import (dataset_has_custom_download_fcn,
                    dataset_has_valid_file_url, get_dataset_info)


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"archive member escapes destination: {member!r}")
    return target


def _extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                _safe_target(destination, item.filename)
            stream.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as stream:
            for item in stream.getmembers():
                _safe_target(destination, item.name)
            stream.extractall(destination, filter="data")
        return
    raise ValueError(f"unsupported dataset archive {archive.name!r}")


def download_dataset(name, ask_first=True):
    info, present = get_dataset_info(name)
    destination = Path(mrst_settings("get", "dataDirectory")) / info.name
    if present:
        return destination, True
    if not mrst_settings("get", "allowDL"):
        return destination, False
    if ask_first:
        answer = input(f"Download {info.name} ({info.filesize or 0:.2f} MB)? [Y/n] ")
        if answer.strip().lower() not in ("", "y", "yes"):
            return destination, False
    if dataset_has_custom_download_fcn(info):
        return destination, bool(info.downloadFcn())
    if not dataset_has_valid_file_url(info):
        return destination, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pymrst-dataset-") as work:
        archive = Path(work) / Path(info.fileurl).name
        with urllib.request.urlopen(info.fileurl) as source, archive.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        _extract(archive, destination)
    return destination, destination.is_dir()


downloadDataset = download_dataset

