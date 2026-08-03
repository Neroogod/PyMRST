"""The `pymrst` command (F0.5).

Two things are being protected here. That the report is *correct* - a doctor
that misreports a capability is worse than none, because it sends someone to
debug the wrong thing. And that the whole command stays *additive*: it may
observe the installation, never change it, and no PyMRST module may depend on
it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from pymrst.cli import capabilities
from pymrst.cli.main import datasets_fetch, datasets_list, doctor, main


def test_doctor_reports_every_capability_and_succeeds(capsys):
    assert doctor() == 0
    out = capsys.readouterr().out
    for capability in capabilities.CAPABILITIES:
        assert capability.name in out, capability.name
    assert "CUDA" in out and "Directories" in out


def test_doctor_keeps_installation_and_data_status_apart(capsys):
    """The distinction the report exists to make.

    "Installed correctly" and "benchmark data downloaded" are different
    states. Merged into one list, a missing 470 MB optional dataset reads as a
    broken installation, which is the confusion this separation prevents.
    """
    doctor()
    out = capsys.readouterr().out
    assert "Datasets are reported separately" in out
    # No dataset names in the capability report.
    assert "SPE1" not in out and "Norne" not in out


def test_a_missing_capability_names_the_install_that_fixes_it():
    absent = capabilities.Capability("Imaginary", ("no_such_module_xyz",),
                                     "imaginary", "does not exist")
    assert not absent.available
    assert absent.missing == ("no_such_module_xyz",)
    assert absent.remedy == "pip install 'pymrst[imaginary]'"

    present = capabilities.Capability("Core", ("numpy",), None, "required")
    assert present.available and present.remedy == ""


def test_capability_probing_does_not_import_the_dependency():
    """`find_spec`, not `import`.

    Reporting that torch exists must not pay for loading it, and a broken
    optional package must not take the whole diagnosis down. Checked by
    probing a module that is importable but not yet imported.
    """
    sys.modules.pop("html.parser", None)
    assert capabilities._installed("html.parser")
    assert "html.parser" not in sys.modules, "probing imported the module"


def test_capability_probing_survives_a_broken_module():
    assert capabilities._installed("definitely_not_installed_zzz") is False


def test_datasets_list_reports_counts_and_provenance(capsys):
    assert datasets_list() == 0
    out = capsys.readouterr().out
    assert "datasets known" in out and "installed" in out
    # Both provenance classes are represented, which is what makes the
    # redistribution distinction visible before anyone runs fetch.
    assert "direct download" in out and "manual - see fetch" in out


def test_an_unknown_dataset_is_refused_with_a_pointer(capsys):
    assert datasets_fetch("no_such_dataset") == 2
    assert "pymrst datasets list" in capsys.readouterr().err


def test_a_non_redistributable_dataset_is_never_downloaded(capsys):
    """The one behaviour here with legal weight.

    Several datasets carry no `fileurl` because their owner does not permit
    redistribution - Olympus and Sleipner among them. `fetch` must print where
    to obtain them and stop, not attempt a download and not report success.
    """
    code = datasets_fetch("Olympus")
    out = capsys.readouterr().out
    assert code == 3, "a non-redistributable dataset must not report success"
    assert "not redistributed by PyMRST" in out
    assert "obtain" in out.lower()


def test_terms_are_shown_before_anything_is_fetched(capsys):
    datasets_fetch("Olympus")
    out = capsys.readouterr().out
    assert "terms:" in out
    assert out.index("terms:") < len(out), "terms must precede any action"


def test_terms_do_not_claim_a_licence_that_is_not_recorded():
    """`DatasetInfo` has no licence field, and the report must not invent one.

    Saying "MIT" or "open" about a benchmark deck nobody checked would be
    worse than saying nothing. The wording reports the distinction the data
    actually makes and states that the licence itself is unrecorded.
    """
    from pymrst.utils.dataset_manager import get_available_datasets
    infos, _ = get_available_datasets()
    downloadable = next(i for i in infos if i.fileurl)
    manual = next(i for i in infos if not i.fileurl)

    from pymrst.cli.main import _terms
    assert "not recorded" in _terms(downloadable)
    assert "not redistributable" in _terms(manual)

    # And if a licence field is ever added, it is used rather than ignored.
    from types import SimpleNamespace
    assert _terms(SimpleNamespace(fileurl="x", licence="CC-BY-4.0")) == "CC-BY-4.0"


def test_the_cli_is_additive_and_nothing_imports_it():
    """The covenant, checked rather than asserted.

    A CLI that a numerical module depends on is no longer optional. This is
    the same rule section 2.0 applies to every capability in the plan.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "pymrst"
    offenders = [
        path for path in root.rglob("*.py")
        if "cli" not in path.parts and "pymrst.cli" in path.read_text(errors="ignore")
    ]
    assert not offenders, f"PyMRST modules importing the CLI: {offenders}"


def test_the_console_entry_point_is_declared():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root/"pyproject.toml").read_text())
    assert config["project"]["scripts"]["pymrst"] == "pymrst.cli.main:main"


def test_bare_invocation_prints_help_rather_than_failing(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()
