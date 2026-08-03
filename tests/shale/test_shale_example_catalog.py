from pathlib import Path

from pymrst.modules.shale.examples.contents import shale_example_catalog


def test_native_example_implementations_live_in_mrst_named_public_owners():
    examples = Path(__file__).parents[2] / "pymrst/modules/shale/examples"
    assert not (examples / "_eagle_ford.py").exists()
    assert not (examples / "_mechanism_comparison.py").exists()
    assert (examples / "eagle_ford.py").read_text().count("def ") >= 8
    assert (examples / "sorption.py").read_text().count("def ") >= 6
    edfm = (examples / "eagle_ford_edfm.py").read_text()
    assert edfm.count("def ") >= 3
    assert "setup_eagle_ford_case" not in edfm
    assert "setup_shale_edfm_operators_tpfa" in edfm
    full = (examples / "eagle_ford_full.py").read_text()
    assert full.count("def ") >= 2
    assert "setup_eagle_ford_case" not in full
    assert "setup_pedfm_operators_tpfa" in full
    diffusion = (examples / "diffusion.py").read_text()
    assert diffusion.count("def ") >= 2
    assert "setup_mechanism_comparison" not in diffusion
    assert "diffusion_coefficients" in diffusion
    assert "NatVarsShaleModel" in diffusion
    gangi = (examples / "gangi.py").read_text()
    assert gangi.count("def ") >= 3
    assert "setup_mechanism_comparison" not in gangi
    assert "permeability_multiplier" in gangi
    assert "NatVarsShaleModel" in gangi
    accelerated = (examples / "eagle_ford_accelerated.py").read_text()
    assert accelerated.count("def ") >= 3
    assert "setup_eagle_ford_case" not in accelerated
    assert "setup_pedfm_operators_tpfa" in accelerated
    assert "DiagonalAutoDiffBackend" in accelerated
    edfm_accelerated = (
        examples / "eagle_ford_edfm_accelerated.py"
    ).read_text()
    assert edfm_accelerated.count("def ") >= 3
    assert "setup_eagle_ford_case" not in edfm_accelerated
    assert "setup_shale_edfm_operators_tpfa" in edfm_accelerated
    assert "DiagonalAutoDiffBackend" in edfm_accelerated
    edfm_full = (examples / "eagle_ford_edfm_full.py").read_text()
    assert edfm_full.count("def ") >= 2
    assert "setup_eagle_ford_case" not in edfm_full
    assert "setup_shale_edfm_operators_tpfa" in edfm_full
    assert "full_field=True" in edfm_full


def test_reference_example_catalog_preserves_mrst_order_and_sections():
    catalog = shale_example_catalog()
    assert [item.owner for item in catalog] == [
        "GangiPlot", "LangmuirPlot", "StochasticFracs", "compositional3D",
        "Sorption", "Diffusion", "Gangi", "EagleFord", "EagleFordEDFM",
    ]
    assert catalog[-2].section == catalog[-1].section == "Section 6"
    assert len({item.module for item in catalog}) == len(catalog)
