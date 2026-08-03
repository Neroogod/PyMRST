"""M2.4 analytic registry and foundation qualification tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from pymrst.research_studio.validation import (
    AnalyticRegistry, EvidenceGraph, EvidenceNodeKind, HYDROSTATIC_SPEC,
    diffusion_exact, foundation_analytic_registry,
)
from pymrst.research_studio.validation.analytic.contracts import AnalyticCaseSpec
from pymrst.research_studio.validation.benchmarks.foundation import (
    FLASH_EQUILIBRIUM_SPEC,
)
from scripts.qualify_analytic_foundation import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/analytic_foundation_evidence_graph.json"


def test_canonical_analytic_and_benchmark_package_boundaries():
    assert AnalyticCaseSpec.__module__.endswith("validation.analytic.contracts")
    assert FLASH_EQUILIBRIUM_SPEC.case_id == "peng_robinson_flash_identities"
    validation = ROOT / "pymrst/research_studio/validation"
    assert (validation / "analytic/contracts.py").is_file()
    assert (validation / "benchmarks/foundation.py").is_file()
    assert not (validation / "analytic.py").exists()
    assert not (validation / "analytic_benchmarks.py").exists()


def test_registry_is_explicit_and_rejects_duplicate_identity():
    registry = foundation_analytic_registry()
    assert registry.case_ids == (
        "hydrostatic_tpfa_column", "pressure_diffusion_1d",
        "radial_pressure_bessel_transient", "buckley_leverett_equal_mobility",
        "terzaghi_consolidation", "mandel_cryer_coupling",
        "elasticity_linear_patch", "peng_robinson_flash_identities",
        "manufactured_accumulation_flux_balance",
    )
    empty = AnalyticRegistry()
    empty.register(HYDROSTATIC_SPEC, lambda: registry.run("hydrostatic_tpfa_column"))
    with pytest.raises(ValueError, match="duplicate"):
        empty.register(HYDROSTATIC_SPEC, lambda: registry.run("hydrostatic_tpfa_column"))


def test_diffusion_exact_obeys_boundary_and_late_time_limits():
    assert diffusion_exact([0.0, 100.0], 1.0).tolist() == pytest.approx([1.0, 0.0])
    late = diffusion_exact(np.array([25.0, 50.0, 75.0]), 1e10)
    np.testing.assert_allclose(late, [0.75, 0.5, 0.25], atol=1e-14)


def test_foundation_cases_pass_accuracy_order_and_conservation():
    results = foundation_analytic_registry().run_all()
    assert all(result.passed for result in results.values())
    hydro = results["hydrostatic_tpfa_column"]
    assert max(level.errors["linf_pressure_Pa"] for level in hydro.levels) < 1e-5
    diffusion = results["pressure_diffusion_1d"]
    assert diffusion.observed_orders["l2"] > 1.9
    assert diffusion.levels[-1].errors["l2"] < 2e-4
    assert max(level.invariants["storage_boundary_balance"]
               for level in diffusion.levels) < 1e-10
    radial = results["radial_pressure_bessel_transient"]
    assert radial.observed_orders["l2"] > 1.85
    assert radial.levels[-1].errors["l2"] < 5e-6
    buckley = results["buckley_leverett_equal_mobility"]
    assert buckley.observed_orders["l1"] > 0.7
    assert buckley.levels[-1].errors["shock_position"] < 5e-3
    terzaghi = results["terzaghi_consolidation"]
    assert terzaghi.observed_orders["pressure_linf"] > 0.85
    assert terzaghi.levels[-1].errors["settlement_linf_relative"] < 5e-4
    mandel = results["mandel_cryer_coupling"]
    assert mandel.observed_orders["pressure_linf"] > 0.8
    assert max(level.invariants["centre_peak_pressure_ratio"]
               for level in mandel.levels) > 1.05
    elasticity = results["elasticity_linear_patch"]
    assert max(level.errors["maximum_relative_displacement"]
               for level in elasticity.levels) < 1e-9
    equilibrium = results["peng_robinson_flash_identities"]
    assert max(level.errors["fugacity_log_ratio"]
               for level in equilibrium.levels) < 1e-8
    balance = results["manufactured_accumulation_flux_balance"]
    assert max(level.errors["maximum_residual"] for level in balance.levels) < 1e-20
    assert max(level.errors["maximum_jacobian_error"]
               for level in balance.levels) < 1e-24


def test_analytic_evidence_graph_regenerates_with_independent_edges():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 18 and len(graph.edges) == 39
    assert all(edge.accepted for edge in graph.edges)
    assert all(not edge.shared_kernels for edge in graph.edges)


def test_persisted_foundation_graph_has_analytic_sources():
    payload = json.loads(ARTIFACT.read_text())
    graph = EvidenceGraph.from_dict(payload["graph"])
    nodes = {node.node_id: node for node in graph.nodes}
    assert nodes["analytic_hydrostatic"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_diffusion"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_radial"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_buckley_leverett"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_terzaghi"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_mandel"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_elasticity_patch"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_flash_equilibrium"].kind is EvidenceNodeKind.ANALYTIC
    assert nodes["analytic_accumulation_flux"].kind is EvidenceNodeKind.ANALYTIC
