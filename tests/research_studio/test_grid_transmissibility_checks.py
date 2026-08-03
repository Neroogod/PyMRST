"""Transmissibility and fault findings (M2.11, remaining checks).

Written before the implementation, because the design question here is not how
to compute a ratio - it is what deserves to be reported. Measured on a healthy
Norne mini: interior face transmissibility spans 5e-301 to 3.6e-11 and 493
interior faces sit more than four orders below the median. Those are the
sealed kz of the layer-4 shale, which is geology, not a defect. A plain
threshold reports all 493 and teaches the reader to ignore the check.

So the contract these tests pin is the separation: a face that is sealed
exactly is deliberate, and a face that is merely *small* against its
neighbours is worth a look.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params.rock import make_rock
from pymrst.research_studio.inspection import Severity, validate

from pymrst.research_studio.inspection.transmissibility import (
    fault_findings, transmissibility_findings,
)


def _wired_into_validate(grid, rock) -> bool:
    """Whether `validate` calls these checks, as distinct from their existing.

    The module can be complete and still not reach a user, so the two are
    asserted separately: the checks below exercise the functions directly, and
    one test asserts the wiring.
    """
    return any(f.check in {"transmissibility", "faults"}
               for f in validate(grid, rock).findings)


def _grid(dims=(4, 3, 2)):
    return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))


def _rock(grid, perm=1.0e-13):
    return make_rock(grid, perm=np.full(grid.cells.num, perm),
                     poro=np.full(grid.cells.num, 0.2))


def test_a_uniform_grid_reports_no_transmissibility_finding():
    grid = _grid()
    assert transmissibility_findings(grid, _rock(grid)) == []


def test_a_sealed_face_is_not_reported_as_an_anomaly():
    """Zero permeability is a decision, not a defect.

    A shale layer with no vertical permeability produces exactly-zero
    transmissibility, and reporting it as suspicious is what makes the check
    unusable on a real field model.
    """
    grid = _grid()
    rock = _rock(grid)
    rock.perm = rock.perm.copy()
    rock.perm[: grid.cells.num // 2] = 0.0

    anomalies = [f for f in transmissibility_findings(grid, rock)
                 if f.severity is not Severity.INFO]
    assert not anomalies, f"a sealed region was reported as anomalous: {anomalies}"
    assert transmissibility_findings(grid, rock), "the seal was not reported at all"


def test_a_single_near_zero_face_among_open_neighbours_is_reported():
    """The case worth catching: small, but not deliberately zero.

    One cell orders of magnitude tighter than everything around it is usually
    a geometry or upscaling accident rather than a seal.
    """
    grid = _grid()
    rock = _rock(grid)
    rock.perm = rock.perm.copy()
    rock.perm[5] = 1.0e-13*1e-9

    findings = transmissibility_findings(grid, rock)
    assert findings, "an isolated near-zero cell was not reported"
    assert any(f.severity is Severity.WARNING for f in findings)
    assert any(f.faces for f in findings), "no face was named"


def test_fault_faces_are_counted_rather_than_flagged():
    """A fault is a feature. Reporting how many there are is useful; calling
    them problems is not."""
    grid = _grid()
    grid.faces.tag = np.zeros(grid.faces.num, dtype=np.int64)
    grid.faces.tag[:4] = 1

    findings = fault_findings(grid)
    assert findings, "tagged fault faces were not reported at all"
    assert all(f.severity is Severity.INFO for f in findings)
    assert "4" in findings[0].message


def test_a_grid_with_no_fault_tag_reports_nothing_rather_than_failing():
    grid = _grid()
    grid.faces.tag = None
    assert fault_findings(grid) == []


def test_the_checks_are_wired_into_validate():
    """Existing and reachable are different claims.

    Every test above calls the functions directly, so all of them would pass
    with a module nothing invokes.
    """
    grid = _grid()
    rock = _rock(grid)
    rock.perm = rock.perm.copy()
    rock.perm[5] = 1.0e-13*1e-9
    assert _wired_into_validate(grid, rock), (
        "transmissibility findings do not reach validate()"
    )
