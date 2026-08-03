"""Promote existing MRST fixture validators into the regular pytest gate."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "examples" / "validation"


def _load(name: str):
    path = VALIDATION / f"validate_phase{name}.py"
    spec = spec_from_file_location(f"ad_mechanics_validate_phase{name}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_file(stem: str):
    path = VALIDATION / f"{stem}.py"
    spec = spec_from_file_location(f"ad_mechanics_{stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("phase", ["62", "63", "64", "65", "66", "67"])
def test_run2d_workflow_matches_matlab_mrst(phase):
    assert _load(phase).main() == 0


def test_adjoint_matches_central_and_control_perturbation_gradients():
    assert _load("360").main() == 0


def test_norne_field_workflow_matches_mrst_and_corrected_physics():
    assert _load_file("validate_norne").main() == 0


def test_norne_matrix_matlab_fixtures_cover_all_six_cases():
    fixture_dir = VALIDATION / "ad_mechanics" / "norne_matrix"
    fixtures = sorted(fixture_dir.glob("mrst_*.mat"))
    assert len(fixtures) == 6
    required = {
        "p_mean", "p_max", "sw_mean", "sw_max", "sg_mean", "sg_max",
        "uplift_max", "displacement_inf", "bhp", "qws", "qos", "qgs",
        "fluid_model", "method", "mrst_version", "matlab_version",
    }
    for fixture in fixtures:
        data = loadmat(fixture, squeeze_me=True)
        assert required <= data.keys()
        assert data["p_mean"].shape == (21,)
        assert data["bhp"].shape == (21, 2)
