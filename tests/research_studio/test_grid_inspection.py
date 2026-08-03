"""Read-only grid interrogation (M2.11).

Each check is exercised against a grid carrying the specific defect it looks
for, because a validator that reports nothing on a healthy grid and nothing on
a broken one is indistinguishable from one that works.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params.rock import make_rock
from pymrst.research_studio.inspection import (
    Severity, explain_connection, inspect_cell, validate,
)


def _grid(dims=(4, 3, 1)):
    return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))


def _rock(grid, poro=0.2):
    return make_rock(grid, perm=np.full(grid.cells.num, 1.0e-13),
                     poro=np.full(grid.cells.num, poro))


def _checks(report) -> set[str]:
    return {finding.check for finding in report.findings}


def _severed_grid():
    """A genuinely disconnected grid: 4x1x1 with cell 1 deactivated.

    Leaves an isolated cell and a connected pair - two regions - with
    `cells.faces` and `faces.neighbors` still agreeing, which hand-editing the
    neighbour table cannot achieve.
    """
    actnum = np.ones(4, dtype=np.int64)
    actnum[1] = 0
    return _build(_corner_point(4, 1, 1, lambda i, j, k, top, ii, jj:
                                float(k + top), actnum=actnum))


def test_a_healthy_grid_reports_nothing():
    """The baseline that makes every other test meaningful."""
    grid = _grid()
    report = validate(grid, _rock(grid))
    assert report.findings == ()
    assert report.ok and bool(report)


def test_a_negative_volume_is_an_error_naming_the_cell():
    grid = _grid()
    grid.cells.volumes = grid.cells.volumes.copy()
    grid.cells.volumes[5] = -1.0

    report = validate(grid)
    assert not report.ok
    finding = next(f for f in report.errors if f.check == "volumes")
    assert finding.cells == (5,)
    assert "non-positive volume" in finding.message


def test_a_vanishing_volume_is_a_warning_not_an_error():
    """The severity distinction, which is the point of the report.

    A negative volume means the discretization cannot be trusted. A very small
    one is legal - it just dominates the timestep. Reporting both as errors
    would make the first invisible among the second.
    """
    grid = _grid()
    grid.cells.volumes = grid.cells.volumes.copy()
    grid.cells.volumes[2] = grid.cells.volumes.max()*1e-9

    report = validate(grid)
    assert report.ok, "a small cell does not invalidate the discretization"
    assert any(f.severity is Severity.WARNING and f.cells == (2,)
               for f in report.findings)


def test_disconnected_regions_are_reported_with_their_sizes():
    """The finding that explains a well doing nothing.

    Flow cannot cross between regions, so a well in the small one drains a
    pocket. Nothing about the grid is invalid, which is why it is a warning
    and why it is easy to miss without being told.

    The grid is disconnected by deactivating a cell, which is how a real model
    ends up this way. Blanking entries in `faces.neighbors` by hand - what this
    test used to do - leaves `cells.faces` pointing at faces that list no
    owner, which is not a disconnected grid but an impossible one, and the
    consistency check now says so.
    """
    grid = _severed_grid()

    report = validate(grid)
    assert report.ok, [str(f) for f in report.findings]
    assert "connectivity" in _checks(report)
    regions = next(f for f in report.findings
                   if f.check == "connectivity" and "regions" in f.message)
    assert "2 disconnected" in regions.message
    assert any("no interior connection" in f.message for f in report.findings)


def test_a_face_joining_a_cell_to_itself_is_an_error():
    grid = _grid()
    neighbors = np.asarray(grid.faces.neighbors).copy()
    interior = np.flatnonzero(np.all(neighbors >= 0, axis=1))
    neighbors[interior[0], 1] = neighbors[interior[0], 0]
    grid.faces.neighbors = neighbors

    report = validate(grid)
    assert not report.ok
    assert any("connect a cell to itself" in f.message for f in report.errors)


def test_porosity_outside_the_unit_interval_is_an_error():
    grid = _grid()
    rock = _rock(grid)
    rock.poro = rock.poro.copy()
    rock.poro[1] = 1.4

    report = validate(grid, rock)
    assert not report.ok
    finding = next(f for f in report.errors if f.check == "rock")
    assert finding.cells == (1,)


def test_a_well_outside_the_grid_or_in_a_dead_cell_is_an_error():
    grid = _grid()
    outside = validate(grid, None, [SimpleNamespace(name="INJ",
                                                    cells=np.array([5, 999]))])
    assert not outside.ok
    assert any("outside the grid" in f.message for f in outside.errors)

    grid.cells.volumes = grid.cells.volumes.copy()
    grid.cells.volumes[3] = 0.0
    dead = validate(grid, None, [SimpleNamespace(name="PROD",
                                                 cells=np.array([3]))])
    assert any("no volume" in f.message for f in dead.errors)


def test_a_well_with_no_perforations_is_reported_but_not_fatal():
    grid = _grid()
    report = validate(grid, None, [SimpleNamespace(name="IDLE",
                                                   cells=np.array([], dtype=int))])
    assert report.ok
    assert any("perforates no cells" in f.message for f in report.warnings)


def test_findings_are_ordered_worst_first():
    """A report read top-down must lead with what breaks the run.

    The defects are chosen so the *natural* append order is wrong: volumes are
    checked before rock, so a small-cell warning is produced before a porosity
    error. Two defects from the same check would come out ordered by accident
    and prove nothing - which is what the first version of this test did, and
    why removing the sort survived it.
    """
    grid = _grid()
    grid.cells.volumes = grid.cells.volumes.copy()
    grid.cells.volumes[2] = grid.cells.volumes.max()*1e-9      # warning, first
    rock = _rock(grid)
    rock.poro = rock.poro.copy()
    rock.poro[1] = 1.4                                          # error, later

    findings = validate(grid, rock).findings
    severities = [f.severity for f in findings]
    assert Severity.ERROR in severities and Severity.WARNING in severities
    assert severities[0] is Severity.ERROR, (
        f"report leads with {severities[0]}, not the error"
    )
    assert severities == sorted(severities, key=lambda s: {
        Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}[s])


def test_explain_connection_names_the_shared_faces():
    grid = _grid()
    link = explain_connection(grid, 0, 1)
    assert link.connected and len(link.faces) == 1
    assert link.areas[0] > 0.0

    neighbors = np.asarray(grid.faces.neighbors)
    face = link.faces[0]
    assert set(int(c) for c in neighbors[face]) == {0, 1}


def test_explain_connection_gives_a_reason_rather_than_a_negative():
    """"Not connected" is unhelpful when the question is why.

    Sharing no face is a different situation from lying in a separate flow
    region, and the second is the one that surprises people.
    """
    grid = _grid()
    distant = explain_connection(grid, 0, 11)
    assert not distant.connected
    assert "share no face" in distant.reason

    neighbors = np.asarray(grid.faces.neighbors).copy()
    neighbors[np.any(neighbors == 7, axis=1)] = -1
    grid.faces.neighbors = neighbors
    severed = explain_connection(grid, 0, 7)
    assert not severed.connected
    assert "different flow regions" in severed.reason


def test_explain_connection_reports_transmissibility_when_given_it():
    grid = _grid()
    transmissibility = np.arange(grid.faces.num, dtype=float)
    link = explain_connection(grid, 0, 1, transmissibility=transmissibility)
    assert link.transmissibilities == (float(link.faces[0]),)


def test_explain_connection_refuses_nonsense_rather_than_guessing():
    grid = _grid()
    assert not explain_connection(grid, 0, 10_000).connected
    assert "outside the grid" in explain_connection(grid, 0, 10_000).reason
    assert "not connected to itself" in explain_connection(grid, 3, 3).reason


def test_inspect_cell_reports_geometry_neighbours_and_properties():
    grid = _grid()
    rock = _rock(grid)
    state = {"pressure": np.full(grid.cells.num, 2.7e7),
             "sW": np.linspace(0.1, 0.9, grid.cells.num)}

    cell = inspect_cell(grid, 5, rock, state)
    assert cell.index == 5
    assert cell.volume > 0.0
    assert len(cell.centroid) == 3
    assert cell.properties["porosity"] == pytest.approx(0.2)
    assert cell.properties["pressure"] == pytest.approx(2.7e7)

    # Neighbours must agree with the connectivity, not be recomputed
    # differently: every one shares a face.
    for other in cell.neighbours:
        assert explain_connection(grid, 5, other).connected


def test_inspect_cell_refuses_an_index_outside_the_grid():
    with pytest.raises(IndexError, match="outside the grid"):
        inspect_cell(_grid(), 10_000)


def test_inspection_never_mutates_the_grid_it_reads():
    """The covenant, at the level of a single call.

    A validator that quietly repairs its input has changed the model. This
    compares the arrays before and after rather than trusting the code to be
    read-only.
    """
    grid = _grid()
    rock = _rock(grid)
    before = {
        "volumes": np.asarray(grid.cells.volumes).copy(),
        "neighbors": np.asarray(grid.faces.neighbors).copy(),
        "centroids": np.asarray(grid.cells.centroids).copy(),
        "poro": np.asarray(rock.poro).copy(),
    }
    validate(grid, rock, [SimpleNamespace(name="W", cells=np.array([0, 1]))])
    explain_connection(grid, 0, 1)
    inspect_cell(grid, 0, rock)

    np.testing.assert_array_equal(np.asarray(grid.cells.volumes), before["volumes"])
    np.testing.assert_array_equal(np.asarray(grid.faces.neighbors), before["neighbors"])
    np.testing.assert_array_equal(np.asarray(grid.cells.centroids), before["centroids"])
    np.testing.assert_array_equal(np.asarray(rock.poro), before["poro"])


def test_inspection_is_additive_and_unimported_by_the_replica_layer():
    root = Path(__file__).resolve().parents[2]/"pymrst"
    offenders = [
        path for path in (root/"modules").rglob("*.py")
        if "research_studio" in path.read_text(errors="ignore")
    ]
    assert not offenders, f"replica modules importing the extension: {offenders}"


# --------------------------------------------------------------------------
# Twisted cells and face orientation (M2.11 remainder)

def _corner_point(nx, ny, nz, depth, actnum=None):
    """A corner-point box whose per-corner depth `depth(i,j,k,top,ii,jj)` we
    control, so a specific cell can be twisted on purpose."""
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord += [float(i), float(j), 0.0, float(i), float(j), 10.0]
    zcorn = []
    for k in range(nz):
        for top in (0, 1):
            for j in range(ny):
                for jj in range(2):
                    for i in range(nx):
                        for ii in range(2):
                            zcorn.append(depth(i, j, k, top, ii, jj))
    return {"cartDims": np.array([nx, ny, nz]),
            "COORD": np.array(coord, dtype=float),
            "ZCORN": np.array(zcorn, dtype=float),
            "ACTNUM": (np.ones(nx*ny*nz, dtype=np.int64) if actnum is None
                       else np.asarray(actnum, dtype=np.int64))}


def _build(deck):
    import warnings as _w
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return compute_geometry(process_grdecl(deck, split_disconnected=False))


def _flat_box():
    return _build(_corner_point(3, 3, 2, lambda i, j, k, top, ii, jj: float(k + top)))


def _twisted_box():
    """Cell (1,1,0) has one top corner driven far below the cell's own base,
    which puts its centroid on the wrong side of two of its faces."""
    def depth(i, j, k, top, ii, jj):
        if (i, j, k) == (1, 1, 0) and top == 1 and (ii, jj) == (0, 0):
            return -20.0
        return float(k + top)
    return _build(_corner_point(3, 3, 2, depth))


def test_a_twisted_cell_is_reported_as_an_error():
    from pymrst.research_studio.inspection import Severity, twist_findings
    findings = twist_findings(_twisted_box())
    errors = [f for f in findings if f.severity is Severity.ERROR]
    assert len(errors) == 1, [str(f) for f in findings]
    assert errors[0].cells, "the finding must name the cell so it can be located"


def test_the_twist_finding_predicts_the_negative_transmissibility():
    """The check is not merely *like* the symptom - it is the same quantity
    with the permeability removed, so it must name the same cells.

    Without this the check could fire on a grid compute_trans is happy with,
    or stay silent on one it repairs, and either way it would not be evidence.
    """
    import warnings as _w
    from pymrst.params.rock import make_rock
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.research_studio.inspection import Severity, twist_findings

    grid = _twisted_box()
    rock = make_rock(grid, perm=np.full((grid.cells.num, 3), 1e-13),
                     poro=np.full(grid.cells.num, 0.2))
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        raw = compute_trans(grid, rock, fix_negative=False)
    negative_cells = set(grid.cell_no()[np.flatnonzero(raw < 0)].tolist())
    assert negative_cells, "the fixture must actually produce negative half-trans"

    reported = {c for f in twist_findings(grid)
                if f.severity is Severity.ERROR for c in f.cells}
    assert reported == negative_cells


def test_compute_trans_hides_the_defect_by_default():
    """Why this check exists at all, asserted rather than claimed in prose.

    MRST's ported default replaces a negative transmissibility with its
    absolute value and warns to stderr. The caller's array is then clean.
    """
    import warnings as _w
    from pymrst.params.rock import make_rock
    from pymrst.solvers.compute_trans import compute_trans

    grid = _twisted_box()
    rock = make_rock(grid, perm=np.full((grid.cells.num, 3), 1e-13),
                     poro=np.full(grid.cells.num, 0.2))
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        raw = compute_trans(grid, rock, fix_negative=False)
        fixed = compute_trans(grid, rock)
    assert (raw < 0).any() and (fixed >= 0).all()
    np.testing.assert_allclose(fixed, np.abs(raw))


def test_a_healthy_corner_point_grid_reports_no_geometry_finding():
    from pymrst.research_studio.inspection import twist_findings
    assert twist_findings(_flat_box()) == []


def test_norne_is_silent_on_both_geometry_checks():
    """The threshold rule, held to a real grid.

    Norne mini's most oblique half-face is at cos 0.489 - about 61 degrees, and
    entirely legitimate corner-point geometry. A skew threshold set anywhere
    near it would report dozens of findings on a correct grid and train the
    reader to ignore the check, which is the failure the transmissibility
    check already had to be rewritten to avoid.
    """
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.modules.deckformat import read_norne_grdecl
    from pymrst.research_studio.inspection import (
        orientation_findings, twist_findings,
    )
    grid = compute_geometry(
        process_grdecl(read_norne_grdecl(subset="mini"), split_disconnected=False))
    assert twist_findings(grid) == []
    assert orientation_findings(grid) == []


def test_a_reversed_face_normal_is_reported():
    from pymrst.research_studio.inspection import Severity, orientation_findings
    grid = _flat_box()
    assert orientation_findings(grid) == []

    neighbors = np.asarray(grid.faces.neighbors)
    interior = np.flatnonzero((neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0))
    grid.faces.normals[interior[3]] *= -1.0

    findings = orientation_findings(grid)
    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert findings[0].faces == (int(interior[3]),)


def test_validate_carries_both_checks():
    """They must reach the report, not only be importable."""
    from pymrst.research_studio.inspection import Severity, validate
    report = validate(_twisted_box())
    geometry = [f for f in report.findings
                if f.check == "geometry" and f.severity is Severity.ERROR]
    assert geometry, [str(f) for f in report.findings]
    assert not report.ok


def test_an_inconsistent_grid_is_named_as_such_not_as_a_twist():
    """The failure this check nearly reported wrongly.

    A grid whose `cells.faces` and `faces.neighbors` disagree makes
    compute_trans take its `else` branch and flip the normal, so every
    projection computed from it has an untrustworthy sign. Calling that a
    twisted cell would send the reader to inspect geometry that is fine.
    """
    from pymrst.research_studio.inspection import Severity, twist_findings
    grid = _flat_box()
    assert twist_findings(grid) == []

    neighbors = np.asarray(grid.faces.neighbors).copy()
    neighbors[np.any(neighbors == 4, axis=1)] = -1
    grid.faces.neighbors = neighbors

    findings = twist_findings(grid)
    assert len(findings) == 1
    assert findings[0].severity is Severity.ERROR
    assert "does not list them as a neighbour" in findings[0].message
    assert "twist" not in findings[0].message.lower()
