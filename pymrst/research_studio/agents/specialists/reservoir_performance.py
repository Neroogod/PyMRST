"""Reservoir, well, region, group, and field performance specialist."""

from . import DeterministicSpecialist
from ..schemas.findings import FindingSeverity


class ReservoirPerformanceSpecialist(DeterministicSpecialist):
    name = "reservoir_performance"
    tool_name = "analyze_reservoir_performance"

    def analyze(self, result):
        findings = list(super().analyze(result))
        pressure = result.value.get("fields", {}).get("pressure")
        if pressure is not None:
            findings.append(self.make_finding(
                result, category="performance.pressure", severity=FindingSeverity.INFO,
                statement=(
                    f"Mean reservoir pressure changed by {pressure['mean_change']:.6g} "
                    f"over the saved observation window."
                ),
                affected_objects=("field:pressure",), ordinal=10_000,
            ))
        production = result.value.get("production")
        if production is not None:
            names = ("oil_produced", "water_produced", "gas_produced")
            available = [f"{name}={production[name]:.6g}" for name in names
                         if name in production]
            if available:
                findings.append(self.make_finding(
                    result, category="performance.production",
                    severity=FindingSeverity.INFO,
                    statement=(
                        "Integrated production over the saved observation window: "
                        + ", ".join(available) + "."
                    ),
                    affected_objects=("field:production",), ordinal=10_001,
                ))
        return tuple(findings)


__all__ = ["ReservoirPerformanceSpecialist"]
