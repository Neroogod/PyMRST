"""Stored-run numerical and reservoir-performance interpretation tests."""

from pathlib import Path

import numpy as np
import pytest

from pymrst.research_studio.agents import ToolCall
from pymrst.research_studio.agents.specialists.economics import EconomicsSpecialist
from pymrst.research_studio.agents.specialists.history_matching import (
    HistoryMatchingSpecialist,
)
from pymrst.research_studio.agents.specialists.numerical_diagnostics import (
    NumericalDiagnosticsSpecialist,
)
from pymrst.research_studio.agents.specialists.reservoir_performance import (
    ReservoirPerformanceSpecialist,
)
from pymrst.research_studio.agents.tools.result_tools import (
    EconomicsTool,
    HistoryMatchTool,
    NumericalDiagnosticsTool,
    ReservoirPerformanceTool,
    ResultAnalysisCatalog,
    ResultAnalysisError,
)
from pymrst.research_studio.experiments import (
    MemberPayload,
    MemberResultStore,
    VariantCoordinate,
)
from pymrst.research_studio.studio import ResultCatalog


def _coordinate():
    return VariantCoordinate(
        "experiment_run", "member_run", 0, "base", "default", {}, {},
    )


def _store(tmp_path: Path, *, start_at_zero=False):
    coordinate = _coordinate()
    store = MemberResultStore(tmp_path / "member")
    store.begin(coordinate, 42, "test:runner")
    times = np.array([0.0, 1.0, 3.0]) if start_at_zero else np.array([1.0, 2.0, 4.0])
    pressure = np.array([
        [30.0e6, 28.0e6],
        [29.0e6, 27.0e6],
        [27.0e6, 25.0e6],
    ])
    store.commit(
        MemberPayload(
            times=times,
            fields={
                "pressure": pressure,
                "sW": np.array([[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]),
            },
            wells={
                "bhp": np.array([[20.0e6, 35.0e6]] * 3),
                "qOs": np.array([[-10.0, -5.0]] * 3),
                "qWs": np.array([[-1.0, -0.5]] * 3),
                "qGs": np.array([[-100.0, -50.0]] * 3),
            },
            convergence={
                "iterations": np.array([[3.0], [10.0], [16.0]]),
                "cuts": np.array([[0.0], [2.0], [0.0]]),
                "cnv": np.array([[1e-3], [2e-4], [1e-5]]),
            },
            regions={
                "field_average_pressure": np.array([[29.0e6], [28.0e6], [26.0e6]]),
            },
            conservation={"mass_balance": np.array([[0.0], [1e-9], [2e-9]])},
            performance={"simulation_seconds": 12.5, "peak_memory_bytes": 2048.0},
            metadata={
                "case": "synthetic_blackoil",
                "well_names": ["P1", "P2"],
                "rate_sign_convention": "positive_into_reservoir",
            },
        ),
        coordinate=coordinate, seed=42, runner="test:runner",
        elapsed_seconds=12.5,
    )
    return store.path


def _catalog(tmp_path, *, start_at_zero=False):
    store = _store(tmp_path, start_at_zero=start_at_zero)
    results = ResultCatalog((tmp_path,), maximum_json_values=100)
    results.register("result_run", "Synthetic black-oil run", store)
    catalog = ResultAnalysisCatalog(results)
    catalog.register("result_run")
    return catalog, store


def test_numerical_diagnostics_localizes_difficult_steps_and_cuts(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = NumericalDiagnosticsTool(catalog).invoke(ToolCall(
        "diagnostics", "analyze_numerical_diagnostics",
        {"result_id": "result_run", "iteration_warning_threshold": 8,
         "iteration_critical_threshold": 15,
         "convergence_limits": {"cnv": 5e-4}},
    ))
    assert result.value["iterations"]["total"] == 29.0
    assert result.value["iterations"]["maximum"] == 16.0
    assert result.value["cuts"]["total"] == 2.0
    assert result.value["cuts"]["steps_with_cuts"] == 1
    assert result.value["timestep"] == {
        "minimum": 1.0, "maximum": 2.0, "median": 1.0,
    }
    categories = {item["category"] for item in result.value["findings"]}
    assert {
        "numerical.iterations", "numerical.timestep_cuts",
        "numerical.convergence_limit",
    } <= categories
    findings = NumericalDiagnosticsSpecialist().analyze(result)
    assert any(item.category == "numerical.summary" for item in findings)
    assert all(item.evidence[0].checksum == catalog.get("result_run")[0].fingerprint
               for item in findings)


def test_diagnostics_accepts_initial_state_saved_at_time_zero(tmp_path):
    catalog, _ = _catalog(tmp_path, start_at_zero=True)
    result = NumericalDiagnosticsTool(catalog).invoke(ToolCall(
        "time_zero", "analyze_numerical_diagnostics",
        {"result_id": "result_run"},
    ))
    assert not any(item["category"] == "numerical.time"
                   for item in result.value["findings"])
    assert result.value["timestep"]["minimum"] == 1.0
    assert result.value["simulated_time"] == 3.0


def test_reservoir_performance_uses_explicit_sign_and_saved_window(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = ReservoirPerformanceTool(catalog).invoke(ToolCall(
        "performance", "analyze_reservoir_performance",
        {"result_id": "result_run", "rate_sign_convention": "positive_into_reservoir"},
    ))
    pressure = result.value["fields"]["pressure"]
    assert pressure["initial_mean"] == 29.0e6
    assert pressure["final_mean"] == 26.0e6
    assert pressure["mean_depletion"] == 3.0e6
    production = result.value["production"]
    assert production["observation_window"] == (1.0, 4.0)
    assert production["oil_produced"] == pytest.approx(45.0)
    assert production["water_produced"] == pytest.approx(4.5)
    assert production["gas_produced"] == pytest.approx(450.0)
    assert production["final_field_water_cut"] == pytest.approx(1.5 / 16.5)
    assert production["final_field_gor"] == pytest.approx(10.0)
    assert result.value["wells"]["qOs"]["entity_names"] == ("P1", "P2")
    assert result.value["regions"]["field_average_pressure"]["change"] == (-3.0e6,)
    findings = ReservoirPerformanceSpecialist().analyze(result)
    assert {item.category for item in findings} == {
        "performance.pressure", "performance.production",
    }


def test_native_sign_does_not_invent_production_kpis(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = ReservoirPerformanceTool(catalog).invoke(ToolCall(
        "native", "analyze_reservoir_performance",
        {"result_id": "result_run", "rate_sign_convention": "native"},
    ))
    assert result.value["production"] is None
    assert "signed_integral_over_saved_window" in result.value["wells"]["qOs"]


def test_rate_sign_convention_can_come_from_stored_metadata(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = ReservoirPerformanceTool(catalog).invoke(ToolCall(
        "metadata_sign", "analyze_reservoir_performance",
        {"result_id": "result_run"},
    ))
    assert result.value["production"]["sign_convention"] == "positive_into_reservoir"
    assert result.value["production"]["oil_produced"] == pytest.approx(45.0)


def test_result_analysis_detects_post_registration_store_mutation(tmp_path):
    catalog, store = _catalog(tmp_path)
    with (Path(store) / "arrays.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ResultAnalysisError, match="changed after"):
        catalog.get("result_run")


def test_result_analysis_full_verification_is_available(tmp_path):
    catalog, _ = _catalog(tmp_path)
    assert catalog.verify("result_run") is True


def test_economics_prices_only_assigned_phases_and_flags_the_rest(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = EconomicsTool(catalog).invoke(ToolCall(
        "economics", "analyze_economics", {
            "result_id": "result_run", "prices": {"oil": 50.0},
            "rate_sign_convention": "positive_into_reservoir",
        },
    ))
    economics = result.value["economics"]
    # Oil produced 15/time-unit over [1, 4] -> volume 45, revenue 50*45.
    assert economics["produced_volume"]["oil"] == pytest.approx(45.0)
    assert economics["revenue"]["oil"] == pytest.approx(2250.0)
    assert economics["revenue"]["water"] == pytest.approx(0.0)
    assert economics["net_present_value"] == pytest.approx(2250.0)
    assert economics["undiscounted_net_cash_flow"] == pytest.approx(2250.0)
    categories = {item["category"] for item in result.value["findings"]}
    assert "economics.missing_price" in categories
    missing = next(item for item in result.value["findings"]
                   if item["category"] == "economics.missing_price")
    assert "gas" in missing["statement"] and "water" in missing["statement"]
    findings = EconomicsSpecialist().analyze(result)
    assert any(item.category == "economics.summary" for item in findings)
    assert all(item.evidence[0].checksum == catalog.get("result_run")[0].fingerprint
               for item in findings)


def test_economics_discounting_reduces_net_present_value(tmp_path):
    catalog, _ = _catalog(tmp_path)
    undiscounted = EconomicsTool(catalog).invoke(ToolCall(
        "flat", "analyze_economics", {
            "result_id": "result_run", "prices": {"oil": 50.0},
            "rate_sign_convention": "positive_into_reservoir",
        },
    )).value["economics"]["net_present_value"]
    discounted = EconomicsTool(catalog).invoke(ToolCall(
        "discounted", "analyze_economics", {
            "result_id": "result_run", "prices": {"oil": 50.0},
            "discount_rate": 0.1,
            "rate_sign_convention": "positive_into_reservoir",
        },
    )).value["economics"]["net_present_value"]
    times = np.array([1.0, 2.0, 4.0])
    expected_discounted = float(np.trapezoid(750.0*np.exp(-0.1*times), times))
    assert discounted == pytest.approx(expected_discounted)
    assert discounted < undiscounted


def test_economics_capital_cost_can_flip_npv_negative(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = EconomicsTool(catalog).invoke(ToolCall(
        "capex", "analyze_economics", {
            "result_id": "result_run", "prices": {"oil": 50.0},
            "capital_cost": 3000.0,
            "rate_sign_convention": "positive_into_reservoir",
        },
    ))
    assert result.value["economics"]["net_present_value"] == pytest.approx(-750.0)
    categories = {item["category"] for item in result.value["findings"]}
    assert "economics.npv" in categories


def test_economics_under_native_convention_reports_no_cash_flow(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = EconomicsTool(catalog).invoke(ToolCall(
        "native", "analyze_economics",
        {"result_id": "result_run", "rate_sign_convention": "native"},
    ))
    assert result.value["economics"] is None
    assert result.value["findings"] == ()


def test_economics_rejects_unknown_price_phase(tmp_path):
    catalog, _ = _catalog(tmp_path)
    with pytest.raises(ValueError, match="prices keys"):
        EconomicsTool(catalog).invoke(ToolCall(
            "bad", "analyze_economics",
            {"result_id": "result_run", "prices": {"condensate": 1.0}},
        ))


def _pressure_observation(times, values, *, entity_index=0, weight=1.0):
    return {
        "category": "fields", "name": "pressure", "entity_index": entity_index,
        "times": times, "values": values, "weight": weight,
    }


def test_history_match_exact_agreement_has_zero_misfit(tmp_path):
    catalog, _ = _catalog(tmp_path)
    # Piecewise-linear through (1, 30e6), (2, 29e6), (4, 27e6) at cell 0:
    # at t=3, halfway from t=2 to t=4, the value is 29e6 - 1e6 = 28e6.
    result = HistoryMatchTool(catalog).invoke(ToolCall(
        "match", "analyze_history_match", {
            "result_id": "result_run",
            "observations": {
                "field_pressure": _pressure_observation(
                    (1.0, 2.0, 3.0), (30.0e6, 29.0e6, 28.0e6),
                ),
            },
        },
    ))
    summary = result.value["observations"]["field_pressure"]
    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["normalized_rmse"] == pytest.approx(0.0)
    assert result.value["aggregate_normalized_rmse"] == pytest.approx(0.0)
    assert result.value["findings"] == ()
    findings = HistoryMatchingSpecialist().analyze(result)
    assert any(item.category == "history_matching.summary" for item in findings)


def test_history_match_flags_poor_agreement(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = HistoryMatchTool(catalog).invoke(ToolCall(
        "match", "analyze_history_match", {
            "result_id": "result_run",
            "observations": {
                "field_pressure": _pressure_observation(
                    (1.0, 2.0, 3.0), (25.0e6, 24.0e6, 23.0e6),
                ),
            },
        },
    ))
    summary = result.value["observations"]["field_pressure"]
    assert summary["rmse"] == pytest.approx(5.0e6)
    assert summary["normalized_rmse"] == pytest.approx(0.2)
    categories = {item["category"] for item in result.value["findings"]}
    assert "history_matching.misfit" in categories


def test_history_match_reports_missing_quantity_as_a_finding(tmp_path):
    catalog, _ = _catalog(tmp_path)
    result = HistoryMatchTool(catalog).invoke(ToolCall(
        "match", "analyze_history_match", {
            "result_id": "result_run",
            "observations": {
                "ghost": {
                    "category": "wells", "name": "does_not_exist",
                    "times": (1.0,), "values": (1.0,),
                },
            },
        },
    ))
    assert result.value["observations"] == {}
    assert result.value["aggregate_normalized_rmse"] is None
    categories = {item["category"] for item in result.value["findings"]}
    assert "history_matching.missing_quantity" in categories


def test_history_match_rejects_observations_outside_the_saved_window(tmp_path):
    catalog, _ = _catalog(tmp_path)
    with pytest.raises(ResultAnalysisError, match="outside the saved"):
        HistoryMatchTool(catalog).invoke(ToolCall(
            "match", "analyze_history_match", {
                "result_id": "result_run",
                "observations": {
                    "field_pressure": _pressure_observation((0.0,), (30.0e6,)),
                },
            },
        ))


def test_history_match_rejects_non_increasing_observed_times(tmp_path):
    catalog, _ = _catalog(tmp_path)
    with pytest.raises(ValueError, match="strictly increasing"):
        HistoryMatchTool(catalog).invoke(ToolCall(
            "match", "analyze_history_match", {
                "result_id": "result_run",
                "observations": {
                    "field_pressure": _pressure_observation(
                        (2.0, 1.0), (29.0e6, 30.0e6),
                    ),
                },
            },
        ))
