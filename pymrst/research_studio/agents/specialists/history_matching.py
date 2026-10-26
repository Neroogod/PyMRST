"""History-matching objective, parameter, and evidence specialist."""

from . import DeterministicSpecialist
from ..schemas.findings import FindingSeverity


class HistoryMatchingSpecialist(DeterministicSpecialist):
    name = "history_matching"
    tool_name = "analyze_history_match"

    def analyze(self, result):
        findings = list(super().analyze(result))
        aggregate = result.value.get("aggregate_normalized_rmse")
        observations = result.value.get("observations", {})
        if aggregate is not None:
            findings.append(self.make_finding(
                result, category="history_matching.summary",
                severity=FindingSeverity.INFO,
                statement=(
                    f"History match compared {len(observations)} observed "
                    f"quantity/quantities against the stored result with a "
                    f"weighted-average normalized RMSE of {aggregate:.6g}."
                ),
                affected_objects=("field:history_match",), ordinal=10_000,
            ))
        return tuple(findings)


__all__ = ["HistoryMatchingSpecialist"]
