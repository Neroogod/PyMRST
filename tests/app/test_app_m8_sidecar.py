from __future__ import annotations

import os

import pytest

from pymrst.app.desktop_sidecar import _token_from_environment, parser
from pymrst.app.packaging import SUPPORTED_TARGETS, native_target, sidecar_filename


def test_desktop_sidecar_requires_strong_ephemeral_credential(monkeypatch):
    monkeypatch.delenv("PYMRST_DESKTOP_BOOTSTRAP_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="missing or invalid"):
        _token_from_environment()
    monkeypatch.setenv("PYMRST_DESKTOP_BOOTSTRAP_TOKEN", "short")
    with pytest.raises(RuntimeError, match="missing or invalid"):
        _token_from_environment()
    token = "a" * 64
    monkeypatch.setenv("PYMRST_DESKTOP_BOOTSTRAP_TOKEN", token)
    assert _token_from_environment() == token


def test_sidecar_cli_defaults_to_os_selected_port(tmp_path):
    arguments = parser().parse_args(["--workspace", str(tmp_path)])
    assert arguments.workspace == tmp_path
    assert arguments.port == 0


def test_native_sidecar_target_and_names_follow_tauri_convention():
    target = native_target()
    assert target in SUPPORTED_TARGETS
    name = sidecar_filename(target)
    assert name.startswith(f"pymrst-sidecar-{target}")
    assert name.endswith(".exe") is ("windows" in target)


def test_unknown_sidecar_target_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        sidecar_filename("unknown-target")
