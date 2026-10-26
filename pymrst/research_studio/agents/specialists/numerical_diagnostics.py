"""Nonlinear, linear, timestep, convergence, and failure specialist."""

from . import DeterministicSpecialist
from ..schemas.findings import FindingSeverity


class NumericalDiagnosticsSpecialist(DeterministicSpecialist):
    name = "numerical_diagnostics"
    tool_name = "analyze_numerical_diagnostics"

    def analyze(self, result):
        findings = list(super().analyze(result))
        iterations = result.value.get("iterations")
        if iterations is not None:
            statement = (
                f"The run completed {result.value['report_steps']} saved report steps "
                f"with {iterations['total']:.0f} total nonlinear iterations; "
                f"the maximum was {iterations['maximum']:.0f}."
            )
            findings.append(self.make_finding(
                result, category="numerical.summary", severity=FindingSeverity.INFO,
                statement=statement, ordinal=10_000,
            ))
        return tuple(findings)


__all__ = ["NumericalDiagnosticsSpecialist"]
