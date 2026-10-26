import importlib
import json
from pathlib import Path


def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMRST_CONFIG_DIR", str(tmp_path / "config"))
    module = importlib.import_module("pymrst.utils.settings_manager.mrst_settings")
    module._SETTINGS = None
    return module


def test_defaults_create_real_data_and_output_directories(monkeypatch, tmp_path):
    module = fresh_settings(monkeypatch, tmp_path)
    settings = module.mrst_settings("get")
    assert Path(settings["dataDirectory"]).is_dir()
    assert Path(settings["outputDirectory"]).is_dir()


def test_set_is_persistent_and_toggle_works(monkeypatch, tmp_path):
    module = fresh_settings(monkeypatch, tmp_path)
    module.mrst_settings("set", "promptDL", "off")
    assert module.mrst_settings("get", "promptDL") is False
    module.mrst_settings("set", "promptDL", "toggle")
    assert module.mrst_settings("get", "promptDL") is True
    saved = json.loads((tmp_path / "config/settings.json").read_text())
    assert saved["promptDL"] is True


def test_unknown_setting_and_verb_are_rejected(monkeypatch, tmp_path):
    module = fresh_settings(monkeypatch, tmp_path)
    import pytest
    with pytest.raises(KeyError):
        module.mrst_settings("get", "doesNotExist")
    with pytest.raises(ValueError):
        module.mrst_settings("explode")


def test_struct_to_settings_ports_known_values_and_rejects_unknowns(tmp_path):
    from pymrst.utils.settings_manager.private import struct_to_settings
    settings = struct_to_settings({
        "dataDirectory": str(tmp_path / "data"),
        "outputDirectory": str(tmp_path / "output"),
        "allowDL": False,
    })
    assert settings.allowDL is False
    assert Path(settings.dataDirectory).is_dir()
    import pytest
    with pytest.raises(KeyError, match="unsupported"):
        struct_to_settings({"mystery": 1})


def test_override_setting_changes_this_process_only(monkeypatch, tmp_path):
    module = fresh_settings(monkeypatch, tmp_path)
    before = module.mrst_settings("get", "outputDirectory")
    saved_before = (tmp_path / "config/settings.json").read_text() \
        if (tmp_path / "config/settings.json").exists() else None
    with module.override_setting("outputDirectory", str(tmp_path / "elsewhere")):
        assert module.mrst_settings("get", "outputDirectory") == str(tmp_path / "elsewhere")
    assert module.mrst_settings("get", "outputDirectory") == before
    saved_after = (tmp_path / "config/settings.json").read_text() \
        if (tmp_path / "config/settings.json").exists() else None
    assert saved_after == saved_before
