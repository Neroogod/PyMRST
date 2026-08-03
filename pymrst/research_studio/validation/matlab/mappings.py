"""Named mapping contracts shared by MATLAB qualification adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..mappings import EntityMapping, identity_mapping


@dataclass(frozen=True, slots=True)
class MatlabResultMappings:
    mappings: Mapping[str, EntityMapping]

    def __post_init__(self):
        normalized = dict(self.mappings)
        if not normalized or any(not name for name in normalized):
            raise ValueError("MATLAB result mappings require named axes")
        object.__setattr__(self, "mappings", normalized)

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(f"{name}:v1" for name in sorted(self.mappings))

    def to_dict(self):
        return {name: mapping.to_dict() for name, mapping in self.mappings.items()}


def result_mappings(*, report_times: Sequence[float], candidate_times: Sequence[float],
                    reference_cells: Sequence[int], candidate_cells: Sequence[int],
                    reference_wells: Sequence[str], candidate_wells: Sequence[str],
                    reference_phases: Sequence[str] = (),
                    candidate_phases: Sequence[str] = ()) -> MatlabResultMappings:
    mappings = {
        "report_time": identity_mapping(report_times, candidate_times, kind="report_time"),
        "cell": identity_mapping(reference_cells, candidate_cells, kind="cell"),
        "well": identity_mapping(reference_wells, candidate_wells, kind="well"),
    }
    if reference_phases or candidate_phases:
        mappings["phase"] = identity_mapping(
            reference_phases, candidate_phases, kind="phase")
    return MatlabResultMappings(mappings)


__all__ = ["MatlabResultMappings", "result_mappings"]
