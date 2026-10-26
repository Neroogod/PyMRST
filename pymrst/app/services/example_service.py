"""Curated catalog of PyMRST example scripts runnable from the Studio app.

Every entry's ``module``/``function`` is a literal, developer-chosen string —
never derived from user input — matching the same trust boundary
``IsolatedRunManager`` already uses for every other registered runner. A
client only ever supplies an ``example_id``, which is looked up in this
catalog; it can never select arbitrary code to import or call.

Each entry was verified to import and run to completion with ``pyvista``,
``matplotlib``, ``torch``, and the other PyInstaller-excluded packages
genuinely absent (not just unused), matching the packaged desktop sidecar's
actual environment rather than this repository's full development install.
"""

from __future__ import annotations

from ..schemas.example import ExampleDefinition, ExampleRuntimeClass

_FAST = ExampleRuntimeClass.FAST

_CATALOG: tuple[ExampleDefinition, ...] = (
    ExampleDefinition(
        "buckley-leverett-1d", "Buckley-Leverett displacement",
        "Classic 1D two-phase waterflood displacement front, validated "
        "against the analytic Buckley-Leverett solution.",
        "Automatic differentiation",
        "pymrst.modules.ad_core.examples.ad_buckley_leverett_1d",
        "run_ad_buckley_leverett_1d", _FAST,
    ),
    ExampleDefinition(
        "blackoil-gravity-segregation", "Black-oil gravity segregation",
        "Three-phase gravity segregation in a small column, showing fluids "
        "settling by density.",
        "Black-oil",
        "pymrst.modules.ad_blackoil.examples.blackoil_tutorial_grav_seg",
        "run_blackoil_tutorial_grav_seg", _FAST,
    ),
    ExampleDefinition(
        "blackoil-one-phase", "Black-oil one-phase depletion",
        "Single-phase reservoir depletion under an open boundary.",
        "Black-oil",
        "pymrst.modules.ad_blackoil.examples.blackoil_tutorial_one_phase",
        "run_blackoil_tutorial_one_phase", _FAST,
    ),
    ExampleDefinition(
        "immiscible-cfl-waterflood", "Heterogeneous waterflood with CFL diagnostics",
        "1D heterogeneous immiscible waterflood with a CFL-number "
        "diagnostic tracking the saturation front.",
        "Black-oil",
        "pymrst.modules.ad_blackoil.examples.immiscible_time_integration_example",
        "run_immiscible_time_integration_example", _FAST,
    ),
    ExampleDefinition(
        "mandel-poroelasticity", "Mandel poroelasticity benchmark",
        "Classic Mandel consolidation problem, checked against its "
        "closed-form analytic solution.",
        "Geomechanics",
        "pymrst.modules.ad_mechanics.examples.mandel", "run_mandel", _FAST,
    ),
    ExampleDefinition(
        "terzaghi-consolidation", "Terzaghi consolidation benchmark",
        "1D Terzaghi consolidation, checked against its closed-form "
        "analytic solution.",
        "Geomechanics",
        "pymrst.modules.ad_mechanics.examples.terzaghi", "run_terzaghi", _FAST,
    ),
    ExampleDefinition(
        "two-scale-coarse-pressure", "Two-scale coarse-grid pressure solve",
        "Left-to-right pressure drop solved on a fine grid and on a "
        "coarsened upscaled grid, for comparison.",
        "Grids",
        "pymrst.modules.coarsegrid.examples.coarsegrid_example1",
        "two_scale_pressure", _FAST,
    ),
    ExampleDefinition(
        "fault-pinch-nnc-waterflood", "Faulted waterflood with pinch-outs and NNCs",
        "Faulted corner-point model with layer pinch-outs and "
        "non-neighbor connections, run through a two-phase waterflood.",
        "Grids",
        "pymrst.gridprocessing.examples.fault_pinch_nnc_waterflood",
        "run_fault_pinch_nnc_waterflood", _FAST,
    ),
    ExampleDefinition(
        "grid-factory-tour", "Grid factory tour",
        "Builds one grid from each of PyMRST's factory routines: "
        "Cartesian, tensor, tetrahedral, triangular, extruded, "
        "hexahedral, and twisted.",
        "Grids",
        "pymrst.examples.grids.grid_tutorial_intro",
        "build_all_grid_types", _FAST,
    ),
    ExampleDefinition(
        "vem-2d-elasticity", "2D linear elasticity on a mixed grid",
        "Virtual Element Method linear elasticity solve on a mixed "
        "Cartesian/PEBI/triangular grid.",
        "Geomechanics",
        "pymrst.modules.vemmech.examples.example_2d",
        "run_example_2d", _FAST,
    ),
)

_CATALOG_BY_ID = {item.example_id: item for item in _CATALOG}


class ExampleCatalogService:
    """Read-only accessor over the fixed example catalog."""

    def list(self) -> tuple[ExampleDefinition, ...]:
        return _CATALOG

    def get(self, example_id: str) -> ExampleDefinition:
        try:
            return _CATALOG_BY_ID[str(example_id)]
        except KeyError as error:
            raise KeyError(f"unknown example: {example_id}") from error


__all__ = ["ExampleCatalogService"]
