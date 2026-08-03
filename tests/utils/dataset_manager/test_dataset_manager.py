import importlib
import zipfile
from pathlib import Path

import pytest


def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMRST_CONFIG_DIR", str(tmp_path / "config"))
    settings = importlib.import_module("pymrst.utils.settings_manager.mrst_settings")
    settings._SETTINGS = None
    settings.mrst_settings("set", "dataDirectory", str(tmp_path / "data"))


def test_critical_reservoir_datasets_are_discoverable(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from pymrst.utils.dataset_manager import get_available_datasets
    infos, present = get_available_datasets()
    names = {item.name for item in infos}
    assert {"SPE1", "SPE3", "SPE9", "SPE10", "Egg", "Norne",
            "norne_field_model"} <= names
    assert len(infos) == len(present) == 39
    assert {"aquifertest", "BedModel2", "CO2Atlas", "BlackoilPolymer2d",
            "geothermal", "EOR_Book_II"} <= names
    assert {"IGEMSgrids", "IGEMSsample", "Johansen", "MSFractures",
            "Norne_Ensemble", "NPD_Utsira", "Olympus", "SAIGUP"} <= names
    assert {"SFTSPE9", "simplecomp", "Sleipner", "Sleipner2019",
            "SleipnerPlumes", "SPE10_Layer_Compositional"} <= names


def test_metadata_and_presence_follow_the_data_directory(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from pymrst.utils.dataset_manager import get_dataset_info
    info, present = get_dataset_info("SPE1")
    assert info.cells == 300 and info.hasGrid and info.hasRock and info.hasFluid
    assert not present
    (tmp_path / "data/SPE1").mkdir(parents=True)
    assert get_dataset_info("spe1")[1]


def test_get_dataset_path_can_resolve_a_manual_install(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from pymrst.utils.dataset_manager import get_dataset_path
    path = get_dataset_path("Egg", download=False, skip_available_check=True)
    assert path == tmp_path / "data/Egg"


def test_safe_zip_extraction_rejects_path_traversal(tmp_path):
    from pymrst.utils.dataset_manager.download_dataset import _extract
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="escapes destination"):
        _extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_zip_extraction_installs_normal_content(tmp_path):
    from pymrst.utils.dataset_manager.download_dataset import _extract
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("CASE/CASE.DATA", "RUNSPEC\n/")
    destination = tmp_path / "out"
    _extract(archive, destination)
    assert (destination / "CASE/CASE.DATA").read_text() == "RUNSPEC\n/"


def test_list_dataset_examples_parses_module_qualifiers(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from pymrst.utils.dataset_manager.utils.list_dataset_examples import (
        list_dataset_examples,
    )
    examples = list_dataset_examples("SPE1")
    assert examples and examples[0].module == "ad-blackoil"
    assert examples[0].name == "runAndCompareSPE1"


def test_dataset_browser_filters_and_returns_structured_details(monkeypatch, tmp_path):
    configured(monkeypatch, tmp_path)
    from pymrst.utils.dataset_manager.gui import DatasetBrowser
    browser = DatasetBrowser()
    matches = browser.filter("compositional")
    assert any(item.name == "simplecomp" for item, _ in matches)
    details = browser.details("SPE10")
    assert details["cells"] == 1122000 and details["present"] is False
