"""Production economics, constraints, objectives, and decision specialist."""

from . import DeterministicSpecialist
from ..schemas.findings import FindingSeverity


class EconomicsSpecialist(DeterministicSpecialist):
    name = "economics"
    tool_name = "analyze_economics"

    def analyze(self, result):
        findings = list(super().analyze(result))
        economics = result.value.get("economics")
        if economics is not None:
            findings.append(self.make_finding(
                result, category="economics.summary", severity=FindingSeverity.INFO,
                statement=(
                    f"Net present value over the saved observation window is "
                    f"{economics['net_present_value']:.6g} at a discount rate of "
                    f"{economics['discount_rate']:.6g}; the undiscounted net cash "
                    f"flow is {economics['undiscounted_net_cash_flow']:.6g}."
                ),
                affected_objects=("field:economics",), ordinal=10_000,
            ))
        return tuple(findings)


__all__ = ["EconomicsSpecialist"]
