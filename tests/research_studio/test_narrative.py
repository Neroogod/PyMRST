"""Diagnostic narrative (M1.11).

The property that matters is not readability - it is that every claim is
derived from a recorded field. A narrative that reads well and is not grounded
in the evidence is exactly the failure mode this design exists to prevent, and
it is invisible to anyone reading the output.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.research_studio.evidence.failure_analysis import (
    FailureKind, analyze_residual_history,
)
from pymrst.research_studio.evidence.narrative import (
    NARRATIVE_SCHEMA_VERSION, describe, unexplained_kinds,
)

STALLING = [{"CNV": 1e-3}, {"CNV": 9.9e-4}, {"CNV": 9.8e-4},
            {"CNV": 9.75e-4}, {"CNV": 9.7e-4}, {"CNV": 9.69e-4}]
DIVERGING = [{"CNV": 1e-6}, {"CNV": 1e-4}, {"CNV": 1e-1}, {"CNV": 1e3}]


def _analysis(history, **kwargs):
    return analyze_residual_history(history, converged=False, **kwargs)


def test_a_healthy_run_gets_no_narrative():
    """Silence is the correct output for a run that worked.

    Emitting a diagnosis for a successful solve trains a reader to ignore the
    channel, which costs more than the missing text is worth.
    """
    healthy = analyze_residual_history([{"CNV": 1e-3}, {"CNV": 1e-9}],
                                       converged=True)
    assert describe(healthy) is None


def test_the_summary_names_the_classification_and_where_it_happened():
    narrative = describe(_analysis(STALLING), step=42, elapsed=187.3)
    assert narrative is not None
    assert "report step 42" in narrative.summary
    assert "187.3" in narrative.summary
    assert "stall" in narrative.summary


def test_every_statement_is_traceable_to_recorded_evidence():
    """The design constraint, checked rather than trusted.

    Each statement carries the fields it was built from. A statement with no
    evidence is a claim the narrative cannot support, and is reported by
    `check_traceability` rather than shipped.
    """
    for history in (STALLING, DIVERGING):
        narrative = describe(_analysis(history))
        assert narrative.check_traceability() == ()
        for statement in narrative.statements:
            assert statement.evidence, statement.text


def test_the_numbers_in_the_prose_are_the_numbers_in_the_evidence():
    """A statement may not quote a value it did not record.

    This is the specific way a narrative goes wrong while looking right:
    prose that rounds, rescales or invents a figure the analysis never
    produced.
    """
    analysis = _analysis(DIVERGING)
    narrative = describe(analysis)
    for statement in narrative.statements:
        for key, value in statement.evidence.items():
            if isinstance(value, float):
                assert f"{value:.3g}" in statement.text, (key, value, statement.text)
            elif value is not None:
                assert str(value) in statement.text, (key, value, statement.text)


def test_actions_name_the_setting_and_its_current_value():
    """"Reduce the timestep" is advice; a named setting is a place to look."""
    solver = SimpleNamespace(max_iterations=3, use_line_search=False,
                             use_relaxation=False)
    narrative = describe(_analysis(DIVERGING), solver=solver)
    settings = {a.setting: a.current for a in narrative.actions}
    assert "use_line_search" in settings
    assert settings["use_line_search"] is False
    assert any("line search" in a.text for a in narrative.actions)


def test_an_action_omits_a_setting_it_could_not_read():
    """Without a solver there is no current value, so none is claimed."""
    narrative = describe(_analysis(DIVERGING))
    assert all(a.setting is None and a.current is None for a in narrative.actions)
    assert narrative.actions, "advice should still be given without a solver"


def test_every_failure_kind_has_advice_attached():
    """A new classifier must not arrive without a meaning for the reader.

    `FailureKind` is where a new detector is declared; this fails when one is
    added and nobody decided what it implies.
    """
    assert unexplained_kinds() == (), (
        f"classifiers with no advice: {unexplained_kinds()}"
    )


def test_the_advice_matches_the_diagnosis():
    """Stall and divergence call for opposite responses.

    A stalled residual will not improve with more iterations; a diverging one
    needs damping. Emitting the same advice for both would be worse than
    emitting none, because it reads as specific.
    """
    stall = describe(_analysis(STALLING))
    diverge = describe(_analysis(DIVERGING))

    assert any("will not help" in a.text for a in stall.actions)
    assert any("line search" in a.text for a in diverge.actions)
    assert not any("line search" in a.text for a in stall.actions)


def test_control_chattering_is_reported_from_the_control_history():
    analysis = analyze_residual_history(
        [], converged=False, control_history=("bhp", "rate", "bhp", "rate"),
    )
    narrative = describe(analysis)
    assert "control_chattering" in narrative.kinds
    assert any("switch" in s.text for s in narrative.statements)
    assert any("switching" in a.text for a in narrative.actions)


def test_the_narrative_is_machine_readable_as_well_as_prose():
    """An agent or report generator consumes the structure, not the text."""
    narrative = describe(_analysis(STALLING), step=7)
    payload = narrative.to_dict()
    assert payload["schema_version"] == NARRATIVE_SCHEMA_VERSION
    assert payload["kinds"] and payload["summary"]
    assert all({"text", "evidence"} == set(s) for s in payload["statements"])
    assert all({"text", "setting", "current"} == set(a) for a in payload["actions"])

    import json
    json.dumps(payload)          # must survive the evidence log unchanged


def test_rendering_is_a_view_and_carries_no_information_of_its_own():
    """The prose is generated from the structure, so anything in the text is
    in the dictionary. That is what allows a language model to rephrase the
    output without becoming a source of claims."""
    narrative = describe(_analysis(DIVERGING), step=3)
    rendered = str(narrative)
    assert narrative.summary in rendered
    for statement in narrative.statements:
        assert statement.text in rendered
    for action in narrative.actions:
        assert action.text in rendered


def test_a_real_induced_failure_produces_a_grounded_narrative():
    """End to end on a real solve rather than a hand-written history.

    The synthetic histories above exercise the mapping; this checks the whole
    path from a starved solver through classification to narrative, which is
    where a field name changes and nothing notices.
    """
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case

    case = run_2d_case(fluid_model="water", method="fully_coupled",
                       nx=20, ny=4, nsteps=3)
    solver = NonLinearSolver(max_iterations=1)
    _, report = solver.solve_timestep(case.model, case.states[0],
                                      float(np.asarray(case.dts)[0])*500.0)
    analysis = analyze_residual_history([dict(r) for r in report.residuals],
                                        converged=bool(report.converged))
    narrative = describe(analysis, solver=solver, step=0)
    if narrative is None:
        pytest.skip("this solve converged; nothing to narrate")
    assert narrative.check_traceability() == ()
    assert narrative.statements and narrative.kinds
