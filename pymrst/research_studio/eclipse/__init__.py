"""ECLIPSE binary interoperability (M2.6). Additive: nothing imports this."""

from pymrst.research_studio.eclipse.case import (
    EclipseCase, keyword_support, read_eclipse_case,
)
from pymrst.research_studio.eclipse.export_case import (
    ExportedCase, RunnableWaterDeck, UnsupportedForExport, export_support,
    write_eclipse_case, write_runnable_water_deck,
)
from pymrst.research_studio.eclipse.coverage import (
    AVAILABLE_SEPARATELY, NOT_IMPLEMENTED, READ_BUT_UNUSED, UNRECOGNISED,
    USED, KeywordCoverage,
    case_coverage, classify_keyword, describe_gaps, keyword_coverage,
)
from pymrst.research_studio.eclipse.egrid import (
    CornerPointGrid, read_egrid, write_egrid,
)
from pymrst.research_studio.eclipse.nnc import (
    NonNeighbourConnections, read_nnc, write_nnc,
)
from pymrst.research_studio.eclipse.records import (
    read_keywords, read_records, write_records,
)
from pymrst.research_studio.eclipse.solution import (
    InitFile, RestartStep, read_init, read_restart, restart_series,
)

__all__ = [
    # classification constants
    "AVAILABLE_SEPARATELY", "NOT_IMPLEMENTED", "READ_BUT_UNUSED",
    "UNRECOGNISED", "USED",
    # types
    "CornerPointGrid", "EclipseCase", "ExportedCase", "InitFile",
    "KeywordCoverage", "NonNeighbourConnections", "RestartStep",
    "RunnableWaterDeck", "UnsupportedForExport",
    # reading
    "case_coverage", "classify_keyword", "describe_gaps", "keyword_coverage",
    "keyword_support", "read_eclipse_case", "read_egrid", "read_init",
    "read_keywords", "read_nnc", "read_records", "read_restart",
    "restart_series",
    # writing
    "export_support", "write_eclipse_case", "write_egrid", "write_nnc",
    "write_records", "write_runnable_water_deck",
]
