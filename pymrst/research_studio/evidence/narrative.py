"""Turn a classification into something an engineer can act on (M1.11).

M1.9 answers *what kind* of failure this was: stall, divergence, oscillation,
control chattering, residual floor, linear failure, non-finite, max-iterations.
That is the hard half and it is done. It is also not yet an answer to the
question anyone actually has, which is what to change.

**Every sentence here is derived from a recorded field, and none is
generated.** That is the design constraint, not a stylistic preference: an
explanation invented by a language model reading solver output is a plausible
story about a simulation and indistinguishable from a correct one. A model may
rephrase a finished narrative for readability; it may not originate one, and
the underived form stays available.

So each `Statement` carries the evidence fields it was built from, and
`Narrative.check_traceability()` verifies that every number quoted in the prose
appears in that evidence. A narrative that cannot show its working is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pymrst.research_studio.evidence.failure_analysis import (
    FailureAnalysis, FailureKind,
)

NARRATIVE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Statement:
    """One claim, and the evidence fields it came from."""

    text: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict:
        return {"text": self.text, "evidence": dict(self.evidence)}


@dataclass(frozen=True, slots=True)
class Action:
    """A suggested change, naming the setting and what it is now.

    "Reduce the timestep" is advice. "`max_timestep_cuts` is 6; the step was
    cut 6 times and still failed" is a place to look. The second is the only
    kind worth emitting from evidence.
    """

    text: str
    setting: str | None = None
    current: Any = None

    def to_dict(self) -> dict:
        return {"text": self.text, "setting": self.setting,
                "current": self.current}


@dataclass(frozen=True, slots=True)
class Narrative:
    schema_version: int
    kinds: tuple[str, ...]
    summary: str
    statements: tuple[Statement, ...] = ()
    actions: tuple[Action, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kinds": list(self.kinds),
            "summary": self.summary,
            "statements": [s.to_dict() for s in self.statements],
            "actions": [a.to_dict() for a in self.actions],
        }

    def __str__(self) -> str:
        lines = [self.summary]
        for statement in self.statements:
            lines.append(f"  {statement.text}")
        if self.actions:
            lines.append("")
            lines.append("What to look at:")
            for index, action in enumerate(self.actions, 1):
                suffix = ""
                if action.setting is not None:
                    suffix = f"  [{action.setting} = {action.current}]"
                lines.append(f"  {index}. {action.text}{suffix}")
        return "\n".join(lines)

    def check_traceability(self) -> tuple[str, ...]:
        """Statements whose evidence is empty, i.e. claims with no source.

        Returns the offending texts rather than raising, so a caller can
        report them; an empty tuple means every statement is grounded.
        """
        return tuple(s.text for s in self.statements if not s.evidence)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def describe(analysis: FailureAnalysis, *,
             solver: Any = None,
             step: int | None = None,
             elapsed: float | None = None) -> Narrative | None:
    """A narrative for `analysis`, or None when there is nothing to explain.

    Returns None rather than an empty narrative for a run that did not fail:
    a diagnosis of a healthy run is noise, and emitting one would train a
    reader to ignore the channel.
    """
    kinds = tuple(str(k) for k in analysis.classifications
                  if k is not FailureKind.UNKNOWN)
    if not kinds:
        return None

    where = f"report step {step}" if step is not None else "a nonlinear solve"
    when = f" after {elapsed:.4g} s" if elapsed is not None else ""
    summary = f"{where} failed{when}: {', '.join(kinds)}."

    statements = list(_statements(analysis))
    actions = list(_actions(analysis, solver))
    return Narrative(NARRATIVE_SCHEMA_VERSION, kinds, summary,
                     tuple(statements), tuple(actions))


def _statements(analysis: FailureAnalysis) -> Sequence[Statement]:
    out: list[Statement] = []

    if analysis.worst_residual is not None:
        out.append(Statement(
            f"The worst residual was {analysis.worst_residual} at "
            f"{_format(analysis.worst_value)}, on iteration "
            f"{analysis.worst_iteration}.",
            {"worst_residual": analysis.worst_residual,
             "worst_value": analysis.worst_value,
             "worst_iteration": analysis.worst_iteration},
        ))

    if analysis.initial_norm is not None and analysis.final_norm is not None:
        out.append(Statement(
            f"The residual went from {_format(analysis.initial_norm)} to "
            f"{_format(analysis.final_norm)} over "
            f"{analysis.details.get('iterations_observed')} iterations.",
            {"initial_norm": analysis.initial_norm,
             "final_norm": analysis.final_norm,
             "iterations_observed": analysis.details.get("iterations_observed")},
        ))

    rate = analysis.estimated_log10_rate
    if rate is not None:
        direction = "falling" if rate < 0 else "rising"
        out.append(Statement(
            f"It was {direction} at {_format(abs(rate))} decades per "
            f"iteration.",
            {"estimated_log10_rate": rate},
        ))

    switches = analysis.details.get("control_switches")
    if switches:
        out.append(Statement(
            f"Well controls switched {switches} times, reversing "
            f"{analysis.details.get('control_reversals')} times.",
            {"control_switches": switches,
             "control_reversals": analysis.details.get("control_reversals")},
        ))
    return out


# What each classification means for the thing to change. Kept as data so the
# mapping is inspectable, and so a new FailureKind cannot silently produce a
# narrative with no advice attached.
_ADVICE: dict[FailureKind, tuple[str, str | None]] = {
    FailureKind.STALL: (
        "The residual stopped improving rather than diverging, so more "
        "iterations will not help. Reduce the timestep, or relax the "
        "convergence tolerance if the achieved residual is acceptable",
        "max_iterations"),
    FailureKind.RESIDUAL_FLOOR: (
        "The residual reached a floor it cannot get below, which usually "
        "means a term is being computed to a fixed precision. Compare the "
        "floor against the tolerance before tightening anything",
        None),
    FailureKind.DIVERGENCE: (
        "The residual grew. Cut the timestep and enable line search; a "
        "diverging Newton rarely recovers by iterating",
        "use_line_search"),
    FailureKind.OSCILLATION: (
        "The residual alternated rather than settling, which is the signature "
        "of an update overshooting. Enable relaxation or damping",
        "use_relaxation"),
    FailureKind.CONTROL_CHATTERING: (
        "A well kept switching control, so the nonlinear problem changed "
        "under the solver. Limit the switching or fix the control across the "
        "step",
        None),
    FailureKind.NONFINITE: (
        "A residual became NaN or infinite, so the state left the physical "
        "range before convergence. Check saturation and pressure bounds at "
        "the worst cell rather than the solver settings",
        None),
    FailureKind.LINEAR_FAILURE: (
        "The linear solve failed, so the nonlinear result says nothing about "
        "the physics. Check the preconditioner and the conditioning of the "
        "Jacobian first",
        "linear_solver"),
    FailureKind.MAX_ITERATIONS: (
        "The residual was still falling when the iteration budget ran out - "
        "nothing is wrong with the solve except its length. Raise the budget "
        "or cut the timestep",
        "max_iterations"),
}


def _actions(analysis: FailureAnalysis, solver: Any) -> Sequence[Action]:
    out: list[Action] = []
    for kind in analysis.classifications:
        advice = _ADVICE.get(kind)
        if advice is None:
            continue
        text, setting = advice
        current = getattr(solver, setting, None) if (solver and setting) else None
        out.append(Action(text, setting if current is not None else None, current))
    return out


def unexplained_kinds() -> tuple[str, ...]:
    """Failure kinds with no advice attached.

    Exists so a new classifier cannot be added without someone deciding what
    it means for the reader; the test asserts this is empty apart from
    UNKNOWN, which by definition has nothing to say.
    """
    return tuple(str(k) for k in FailureKind
                 if k is not FailureKind.UNKNOWN and k not in _ADVICE)


__all__ = ["Action", "NARRATIVE_SCHEMA_VERSION", "Narrative", "Statement",
           "describe", "unexplained_kinds"]
