"""ECLIPSE unformatted records and EGRID (M2.6).

Two claims are separated here, because conflating them is how a binary reader
ships wrong.

*Round-trip* shows the reader and writer agree. That is necessary and weak: a
pair sharing a misunderstanding round-trips perfectly, which is the trap the
transport Jacobian's finite-difference check turned out to contain.

*Cross-check* shows the reader agrees with something it did not write - here
the text GRDECL path, which is independently validated against MATLAB. That is
what makes these readers testable at all, since this repository contains no
ECLIPSE binary output to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymrst.research_studio.eclipse import (
    read_egrid, read_keywords, read_records, write_egrid, write_records,
)


def test_every_record_type_round_trips():
    """INTE, REAL, DOUB, LOGI and CHAR, with their storage widths respected."""
    import tempfile
    path = Path(tempfile.mkdtemp())/"CASE.INIT"
    write_records(path, [
        ("INTEHEAD", "INTE", np.arange(95, dtype=np.int64)),
        ("DEPTH", "DOUB", np.linspace(2500.0, 2600.0, 7)),
        ("KEYWORDS", "CHAR", ["WOPR", "WWPR", "FPR"]),
        ("LOGIHEAD", "LOGI", np.array([1, 0, 1])),
    ])
    back = read_keywords(path)
    np.testing.assert_array_equal(back["INTEHEAD"], np.arange(95))
    # DOUB is float64 in the file, so this one is exact.
    np.testing.assert_array_equal(back["DEPTH"], np.linspace(2500.0, 2600.0, 7))
    assert back["KEYWORDS"] == ["WOPR", "WWPR", "FPR"]
    np.testing.assert_array_equal(back["LOGIHEAD"], [True, False, True])


def test_an_array_longer_than_one_block_survives():
    """ECLIPSE splits numeric arrays into 1000-value blocks.

    A reader that assumes one block per keyword truncates every real field
    case at 1000 values and reports no error, so the multi-block path is
    exercised deliberately rather than incidentally.
    """
    import tempfile
    path = Path(tempfile.mkdtemp())/"BIG.INIT"
    values = np.linspace(0.1, 0.3, 2500)
    write_records(path, [("PORO", "REAL", values)])
    back = read_keywords(path)
    assert back["PORO"].size == 2500
    np.testing.assert_allclose(back["PORO"], values, rtol=1e-6)


def test_repeated_keywords_are_all_returned_by_read_records():
    """UNRST repeats its keywords once per report step.

    `read_keywords` keeps the first, which is right for a file where a
    keyword appears once and wrong for a restart file - so anything per-step
    must use `read_records`, and that difference is pinned.
    """
    import tempfile
    path = Path(tempfile.mkdtemp())/"CASE.UNRST"
    write_records(path, [
        ("SEQNUM", "INTE", np.array([0])),
        ("PRESSURE", "REAL", np.array([100.0, 200.0])),
        ("SEQNUM", "INTE", np.array([1])),
        ("PRESSURE", "REAL", np.array([300.0, 400.0])),
    ])
    names = [n for n, _, _ in read_records(path)]
    assert names.count("SEQNUM") == 2 and names.count("PRESSURE") == 2
    np.testing.assert_allclose(read_keywords(path)["PRESSURE"], [100.0, 200.0])


def test_an_over_long_keyword_is_refused():
    import tempfile
    with pytest.raises(ValueError, match="eight characters"):
        write_records(Path(tempfile.mkdtemp())/"X", [("TOOLONGKEYWORD", "INTE", [1])])


def test_a_file_without_gridhead_is_not_read_as_a_grid():
    import tempfile
    path = Path(tempfile.mkdtemp())/"NOT.EGRID"
    write_records(path, [("PORO", "REAL", np.ones(4))])
    with pytest.raises(ValueError, match="GRIDHEAD"):
        read_egrid(path)


def test_inconsistent_array_lengths_are_refused_rather_than_reshaped():
    """A COORD that does not match the declared dimensions is a corrupt file.

    Silently accepting it produces a grid whose geometry is wrong in a way no
    later check would attribute to the reader.
    """
    import tempfile
    path = Path(tempfile.mkdtemp())/"BAD.EGRID"
    write_egrid(path, {"cartDims": [2, 2, 1],
                       "COORD": np.zeros(9*6), "ZCORN": np.zeros(8*4),
                       "ACTNUM": np.ones(4)})
    good = read_egrid(path)
    assert good.cartDims == (2, 2, 1)

    write_records(path, [
        ("GRIDHEAD", "INTE", np.array([1, 2, 2, 1] + [0]*96)),
        ("COORD", "REAL", np.zeros(10)),        # wrong length
        ("ZCORN", "REAL", np.zeros(8*4)),
    ])
    with pytest.raises(ValueError, match="COORD has"):
        read_egrid(path)


def test_egrid_agrees_with_the_validated_text_reader_on_norne():
    """The cross-check, and the only externally grounded claim here.

    Norne's corner-point geometry is read by the text GRDECL path, written as
    an EGRID and read back. Matching arrays show the binary path agrees with a
    reader validated against MATLAB, rather than merely with its own writer.
    """
    import tempfile
    pytest.importorskip("pymrst.modules.deckformat")
    from pymrst.modules.deckformat import read_norne_grdecl

    text = read_norne_grdecl(subset="mini")
    path = Path(tempfile.mkdtemp())/"NORNE.EGRID"
    write_egrid(path, text)
    binary = read_egrid(path)

    assert binary.cartDims == tuple(int(d) for d in np.asarray(text["cartDims"]).ravel())
    np.testing.assert_allclose(binary.coord, np.asarray(text["COORD"]).ravel(), rtol=1e-6)
    np.testing.assert_allclose(binary.zcorn, np.asarray(text["ZCORN"]).ravel(), rtol=1e-6)
    np.testing.assert_array_equal(binary.actnum, np.asarray(text["ACTNUM"]).ravel())


def test_the_round_tripped_grid_processes_to_the_same_topology():
    """Topology exact, geometry to float32 - and the difference is the format.

    EGRID stores COORD and ZCORN as REAL, so a round trip cannot be
    bit-identical and claiming otherwise would be false. What must be exact is
    the topology: cell, face and node counts, the neighbour table and the
    fault tags. Those are what a discretization is built from, and a reader
    that perturbed them would produce a different model rather than a slightly
    less precise one.
    """
    import tempfile
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.modules.deckformat import read_norne_grdecl

    text = read_norne_grdecl(subset="mini")
    path = Path(tempfile.mkdtemp())/"NORNE.EGRID"
    write_egrid(path, text)

    from_text = compute_geometry(process_grdecl(text))
    from_binary = compute_geometry(process_grdecl(read_egrid(path).to_grdecl()))

    assert from_text.cells.num == from_binary.cells.num
    assert from_text.faces.num == from_binary.faces.num
    assert from_text.nodes.num == from_binary.nodes.num
    np.testing.assert_array_equal(np.asarray(from_text.faces.neighbors),
                                  np.asarray(from_binary.faces.neighbors))
    np.testing.assert_array_equal(np.asarray(from_text.faces.tag),
                                  np.asarray(from_binary.faces.tag))

    # Quantified rather than asserted away: float32 pillars give about five
    # significant figures on cell volumes.
    volume_error = np.max(np.abs(from_text.cells.volumes - from_binary.cells.volumes)
                          / np.abs(from_text.cells.volumes))
    assert volume_error < 1e-4, volume_error
    assert volume_error > 0.0, "float32 storage should be measurably lossy"


def test_eclipse_reader_is_additive():
    root = Path(__file__).resolve().parents[2]/"pymrst"
    offenders = [
        path for path in (root/"modules").rglob("*.py")
        if "research_studio" in path.read_text(errors="ignore")
    ]
    assert not offenders, f"replica modules importing the extension: {offenders}"


def _count_fortran_records(path: Path) -> int:
    """Blocks in the file, counted from the record markers themselves.

    Deliberately does not use this package's reader. A writer that emits one
    oversized block and a reader that loops until the header count is
    satisfied agree with each other perfectly - that mutation survived every
    other test here - so the block structure has to be read from the bytes.
    """
    import struct
    blocks = 0
    with path.open("rb") as stream:
        while True:
            lead = stream.read(4)
            if len(lead) < 4:
                return blocks
            size = struct.unpack(">i", lead)[0]
            stream.seek(size, 1)
            if len(stream.read(4)) < 4:
                return blocks
            blocks += 1


def test_long_arrays_are_written_in_the_blocks_the_format_requires():
    """ECLIPSE chunks numeric data at 1000 values per block.

    This is the format, not a preference: a file with one 2500-value block is
    unreadable by tooling that follows the specification, and nothing in a
    round trip through this package's own reader would notice.
    """
    import tempfile
    path = Path(tempfile.mkdtemp())/"BLOCKS.INIT"
    write_records(path, [("PORO", "REAL", np.linspace(0.1, 0.3, 2500))])

    # one header block + ceil(2500/1000) data blocks
    assert _count_fortran_records(path) == 1 + 3, (
        "a 2500-value array must be written as three 1000-value blocks"
    )


def test_character_arrays_are_chunked_at_the_format_limit():
    """CHAR blocks hold 105 entries, which is a different limit again."""
    import tempfile
    path = Path(tempfile.mkdtemp())/"NAMES.SMSPEC"
    write_records(path, [("WGNAMES", "CHAR", [f"W{i:03d}" for i in range(250)])])
    assert _count_fortran_records(path) == 1 + 3, (
        "250 names must be written as three 105-entry blocks"
    )


# --------------------------------------------------------------------------
# INIT and UNRST

def _unrst(path: Path, steps=3, cells=6):
    records = []
    for step in range(steps):
        records += [
            ("SEQNUM", "INTE", np.array([step])),
            ("PRESSURE", "REAL", np.full(cells, 200.0 + 10*step)),
            ("SWAT", "REAL", np.full(cells, 0.2 + 0.1*step)),
        ]
    from pymrst.research_studio.eclipse import write_records as _w
    _w(path, records)
    return path


def test_init_reports_its_arrays_and_cell_count():
    import tempfile
    from pymrst.research_studio.eclipse import read_init, write_records
    path = Path(tempfile.mkdtemp())/"C.INIT"
    write_records(path, [
        ("INTEHEAD", "INTE", np.array([1, 605] + [0]*93, dtype=np.int64)),
        ("PORO", "REAL", np.linspace(0.1, 0.3, 605)),
    ])
    init = read_init(path)
    assert init.cells == 605
    assert "PORO" in init.names
    assert init.get("poro") is not None, "lookup should be case-insensitive"


def test_an_empty_file_is_not_read_as_an_init():
    import tempfile
    from pymrst.research_studio.eclipse import read_init, write_records
    path = Path(tempfile.mkdtemp())/"EMPTY.INIT"
    write_records(path, [])
    with pytest.raises(ValueError, match="no records"):
        read_init(path)


def test_a_restart_file_is_split_into_its_report_steps():
    """The thing a keyword mapping gets wrong.

    UNRST repeats PRESSURE once per step. A reader that builds a mapping keeps
    step zero and silently discards the rest of the simulation, which looks
    like a converged run that never evolved.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_restart
    steps = read_restart(_unrst(Path(tempfile.mkdtemp())/"C.UNRST"))
    assert [s.sequence for s in steps] == [0, 1, 2]
    np.testing.assert_allclose(steps[0].get("PRESSURE"), 200.0)
    np.testing.assert_allclose(steps[2].get("PRESSURE"), 220.0)
    np.testing.assert_allclose(steps[2].get("SWAT"), 0.4)


def test_restart_series_stacks_one_keyword_across_steps():
    import tempfile
    from pymrst.research_studio.eclipse import restart_series
    path = _unrst(Path(tempfile.mkdtemp())/"C.UNRST", steps=4, cells=5)
    series = restart_series(path, "PRESSURE")
    assert series.shape == (4, 5)
    np.testing.assert_allclose(series[:, 0], [200.0, 210.0, 220.0, 230.0])


def test_a_keyword_missing_from_any_step_is_refused_by_name():
    """Returning the steps that do have it would misalign the time axis."""
    import tempfile
    from pymrst.research_studio.eclipse import restart_series
    path = _unrst(Path(tempfile.mkdtemp())/"C.UNRST")
    with pytest.raises(KeyError, match="SGAS"):
        restart_series(path, "SGAS")


def test_ragged_steps_are_refused_rather_than_padded():
    """Differing lengths mean the file mixes grids, or the keyword is not
    cell data. Stacking either way invents values."""
    import tempfile
    from pymrst.research_studio.eclipse import restart_series, write_records
    path = Path(tempfile.mkdtemp())/"RAGGED.UNRST"
    write_records(path, [
        ("SEQNUM", "INTE", np.array([0])),
        ("PRESSURE", "REAL", np.ones(6)),
        ("SEQNUM", "INTE", np.array([1])),
        ("PRESSURE", "REAL", np.ones(4)),
    ])
    with pytest.raises(ValueError, match="differing lengths"):
        restart_series(path, "PRESSURE")


def test_units_are_not_converted_because_they_are_not_declared():
    """INIT carries no unit system; RUNSPEC does.

    Guessing one would silently rescale a whole field, so values come back as
    stored and the caller supplies the system from the deck.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_init, write_records
    path = Path(tempfile.mkdtemp())/"U.INIT"
    stored = np.array([1.0, 2.0, 3.0])
    write_records(path, [("PORV", "REAL", stored)])
    np.testing.assert_allclose(read_init(path).get("PORV"), stored)


def test_a_keyword_present_in_only_some_steps_is_refused_naming_them():
    """The case the explicit check exists for, and the one easy to miss.

    A keyword absent from *every* step raises anyway when the lengths are
    gathered, so a test using one passes without exercising the check - which
    is why removing it survived. Gas appearing partway through a run is the
    real situation: the array exists for later steps and not earlier ones, and
    silently returning the steps that have it shifts the whole time axis.
    """
    import tempfile
    from pymrst.research_studio.eclipse import restart_series, write_records
    path = Path(tempfile.mkdtemp())/"PARTIAL.UNRST"
    write_records(path, [
        ("SEQNUM", "INTE", np.array([0])),
        ("PRESSURE", "REAL", np.ones(4)),
        ("SEQNUM", "INTE", np.array([1])),
        ("PRESSURE", "REAL", np.ones(4)),
        ("SGAS", "REAL", np.full(4, 0.3)),      # gas appears at step 1 only
    ])
    with pytest.raises(KeyError, match=r"\[0\]"):
        restart_series(path, "SGAS")


# --------------------------------------------------------------------------
# Assembling a case

def _box_with_inactive(path: Path, nx=4, ny=2, nz=1, off=(1, 6)):
    """A small grid with some cells switched off.

    Necessary because Norne mini has *no* inactive cells, so it cannot
    exercise the active-cell mapping at all - the very path that misplaces
    properties on a real field model.
    """
    from pymrst.research_studio.eclipse import write_egrid
    n = nx*ny*nz
    actnum = np.ones(n, dtype=np.int64)
    actnum[list(off)] = 0
    pillars = (nx + 1)*(ny + 1)
    coord = np.zeros(6*pillars)
    for p in range(pillars):
        i, j = p % (nx + 1), p//(nx + 1)
        coord[6*p:6*p + 6] = [float(i), float(j), 0.0, float(i), float(j), 1.0]
    zcorn = np.tile(np.repeat([0.0, 1.0], 4*nx*ny), nz).astype(float)
    write_egrid(path.with_suffix(".EGRID"),
                {"cartDims": [nx, ny, nz], "COORD": coord, "ZCORN": zcorn,
                 "ACTNUM": actnum})
    return actnum


def test_arrays_over_cartesian_cells_are_mapped_onto_the_active_ones():
    """The mapping a field model depends on and a full grid cannot test.

    ECLIPSE writes some arrays over active cells and some over the whole box.
    Assuming one convention works perfectly until the first model with
    inactive cells, then puts every property in the wrong place.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case, write_records
    stem = Path(tempfile.mkdtemp())/"BOX"
    actnum = _box_with_inactive(stem)
    active = np.flatnonzero(actnum > 0)

    write_records(stem.with_suffix(".INIT"), [
        ("PORO", "REAL", np.full(active.size, 0.2)),          # active length
        ("PERMX", "REAL", np.arange(actnum.size, dtype=float)*1e-14),  # cartesian
    ])
    case = read_eclipse_case(stem)

    assert case.grid.cells.num == active.size
    np.testing.assert_array_equal(
        (case.rock.perm.ravel()/1e-14).round().astype(int), active,
    )
    np.testing.assert_allclose(case.rock.poro, 0.2)


def test_an_array_of_neither_length_is_refused_by_name():
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case, write_records
    stem = Path(tempfile.mkdtemp())/"BOX"
    _box_with_inactive(stem)
    write_records(stem.with_suffix(".INIT"), [
        ("PORO", "REAL", np.ones(3)), ("PERMX", "REAL", np.ones(3)),
    ])
    with pytest.raises(ValueError, match="PORO|PERMX"):
        read_eclipse_case(stem)


def test_a_case_without_init_or_restart_still_yields_a_grid():
    """Missing inputs are reported, never substituted.

    An invented porosity is worse than an absent one, because it runs.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case
    stem = Path(tempfile.mkdtemp())/"BARE"
    _box_with_inactive(stem)
    case = read_eclipse_case(stem)
    assert case.rock is None and case.states == ()
    assert "grid" in case.source and "rock" not in case.source


def test_a_missing_grid_is_an_error_rather_than_an_empty_case():
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case
    with pytest.raises(FileNotFoundError, match="EGRID"):
        read_eclipse_case(Path(tempfile.mkdtemp())/"NOTHERE")


def test_a_disconnected_grid_is_kept_whole_rather_than_split():
    """PyMRST splits disconnected grids by default; ECLIPSE's is one grid.

    Keeping a single component would silently drop part of the model. The
    disconnection is a finding for `inspection.validate`, not a reason to
    discard cells here.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case
    from pymrst.research_studio.inspection import validate
    stem = Path(tempfile.mkdtemp())/"SPLIT"
    # Switch off a middle column so the box falls into two pieces.
    actnum = _box_with_inactive(stem, nx=5, ny=1, nz=1, off=(2,))
    case = read_eclipse_case(stem)
    assert case.grid.cells.num == int(actnum.sum()), "cells were dropped"

    report = validate(case.grid)
    assert any("disconnected" in f.message for f in report.findings), (
        "the split should be reported by inspection, not hidden by the reader"
    )


def test_restart_states_are_named_the_way_pymrst_names_them():
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case, write_records
    stem = Path(tempfile.mkdtemp())/"BOX"
    actnum = _box_with_inactive(stem)
    n = int(actnum.sum())
    write_records(stem.with_suffix(".UNRST"), [
        ("SEQNUM", "INTE", np.array([0])),
        ("PRESSURE", "REAL", np.full(n, 270.0)),
        ("SWAT", "REAL", np.full(n, 0.25)),
        ("SEQNUM", "INTE", np.array([1])),
        ("PRESSURE", "REAL", np.full(n, 280.0)),
        ("SWAT", "REAL", np.full(n, 0.30)),
    ])
    case = read_eclipse_case(stem)
    assert len(case.states) == 2
    assert sorted(case.states[0]) == ["pressure", "sW"]
    np.testing.assert_allclose(case.states[1]["pressure"], 280.0)


def test_keyword_support_states_what_is_read_ignored_and_refused():
    """The package asks for this, and it is not decoration.

    A reader that silently skips a keyword it does not understand produces a
    case that is quietly not the one on disk.
    """
    from pymrst.research_studio.eclipse import keyword_support
    support = keyword_support()
    assert "COORD" in support["grid"] and "PORO" in support["rock"]
    assert "PRESSURE" in support["state"]
    assert support["not_implemented"], "unimplemented areas must be named"
    assert support["refused"], "the refusal rule must be stated"


# --------------------------------------------------------------------------
# Writing a case out (M3.8, ECLIPSE half)

def _norne_case():
    from pymrst.gridprocessing import compute_geometry, process_grdecl
    from pymrst.modules.deckformat import read_norne_grdecl
    from pymrst.params.rock import make_rock

    text = read_norne_grdecl(subset="mini")
    grid = compute_geometry(process_grdecl(text, split_disconnected=False))
    n = grid.cells.num
    rock = make_rock(
        grid,
        perm=np.column_stack([np.full(n, 1e-13), np.full(n, 1e-13),
                              np.full(n, 1e-14)]),
        poro=np.linspace(0.1, 0.3, n),
    )
    states = [{"pressure": np.full(n, 270e5 + s*5e5),
               "sW": np.full(n, 0.2 + 0.05*s)} for s in range(3)]
    return text, grid, rock, states


def test_a_case_written_out_reads_back_as_the_same_case():
    """The acceptance evidence, and a weaker claim than it looks.

    This shows the writer and reader agree. It does *not* show ECLIPSE would
    accept the file - that needs an external simulator, which this repository
    does not have. The distinction is recorded rather than blurred.
    """
    import tempfile
    from pymrst.research_studio.eclipse import read_eclipse_case, write_eclipse_case
    text, grid, rock, states = _norne_case()
    stem = Path(tempfile.mkdtemp())/"OUT"
    write_eclipse_case(stem, grid, grdecl=text, rock=rock, states=states)

    back = read_eclipse_case(stem)
    assert back.grid.cells.num == grid.cells.num
    np.testing.assert_allclose(back.rock.poro, rock.poro, rtol=1e-6)
    np.testing.assert_allclose(back.rock.perm, rock.perm, rtol=1e-6)
    assert len(back.states) == len(states)
    np.testing.assert_allclose(back.states[2]["pressure"],
                               states[2]["pressure"], rtol=1e-6)


def test_a_state_field_with_no_eclipse_keyword_is_refused():
    """No partial export. A file missing a field is worse than no file,
    because it opens and describes a different model."""
    import tempfile
    from pymrst.research_studio.eclipse import (
        UnsupportedForExport, write_eclipse_case,
    )
    text, grid, _, _ = _norne_case()
    with pytest.raises(UnsupportedForExport, match="temperature"):
        write_eclipse_case(Path(tempfile.mkdtemp())/"X", grid, grdecl=text,
                           states=[{"temperature": np.ones(grid.cells.num)}])


def test_a_rock_missing_permeability_is_refused_rather_than_defaulted():
    import tempfile
    from types import SimpleNamespace
    from pymrst.research_studio.eclipse import (
        UnsupportedForExport, write_eclipse_case,
    )
    text, grid, _, _ = _norne_case()
    with pytest.raises(UnsupportedForExport, match="porosity and permeability"):
        write_eclipse_case(Path(tempfile.mkdtemp())/"X", grid, grdecl=text,
                           rock=SimpleNamespace(poro=np.ones(grid.cells.num),
                                                perm=None))


def test_an_array_of_the_wrong_length_is_refused():
    import tempfile
    from pymrst.research_studio.eclipse import (
        UnsupportedForExport, write_eclipse_case,
    )
    text, grid, rock, _ = _norne_case()
    with pytest.raises(UnsupportedForExport, match="cannot be written"):
        write_eclipse_case(Path(tempfile.mkdtemp())/"X", grid, grdecl=text,
                           states=[{"pressure": np.ones(3)}])


def test_the_corner_point_description_is_required_not_reconstructed():
    """Rebuilding COORD/ZCORN from a processed grid is lossy for faults.

    A silently approximated fault is exactly the error this package exists to
    refuse, so the original description is asked for instead.
    """
    import tempfile
    from pymrst.research_studio.eclipse import (
        UnsupportedForExport, write_eclipse_case,
    )
    _, grid, _, _ = _norne_case()
    with pytest.raises(UnsupportedForExport, match="ZCORN|COORD|cartDims"):
        write_eclipse_case(Path(tempfile.mkdtemp())/"X", grid,
                           grdecl={"cartDims": [11, 11, 5]})


def test_a_case_with_no_rock_or_states_writes_only_a_grid():
    import tempfile
    from pymrst.research_studio.eclipse import write_eclipse_case
    text, grid, _, _ = _norne_case()
    exported = write_eclipse_case(Path(tempfile.mkdtemp())/"BARE", grid,
                                  grdecl=text)
    assert [p.suffix for p in exported.files] == [".EGRID"]


def test_export_support_names_what_is_not_written():
    """The list that decides whether an exported case is runnable elsewhere.

    It is not: a deck needs wells, a schedule and fluid properties, and this
    writes none of them. Saying so is the difference between an honest export
    and one that produces a file nobody can run.
    """
    from pymrst.research_studio.eclipse import export_support
    support = export_support()
    missing = " ".join(support["not_written"]).lower()
    for expected in ("well", "schedule", "pvt"):
        assert expected in missing, f"{expected} should be listed as not written"
    assert support["refused"], "the refusal rules must be stated"


# --------------------------------------------------------------------------
# Keyword coverage: what a given file loses (M2.6 remainder)

def test_a_case_this_package_wrote_reports_complete_coverage():
    """The threshold rule, one more time.

    `read_eclipse_case` is permissive by design, so the coverage report is the
    thing that makes its omissions visible - which only works if it stays quiet
    on a good file. An earlier version counted FILEHEAD and INTEHEAD as dropped
    and marked every valid file incomplete, including the ones written here.
    """
    import tempfile
    from pymrst.research_studio.eclipse import case_coverage, write_eclipse_case
    text, grid, rock, states = _norne_case()
    stem = Path(tempfile.mkdtemp())/"COV"
    write_eclipse_case(stem, grid, grdecl=text, rock=rock, states=states)

    reports = case_coverage(stem)
    assert set(reports) == {"EGRID", "INIT", "UNRST"}
    for kind, report in reports.items():
        assert report.complete, f"{kind}: {describe_gaps_text(report)}"


def describe_gaps_text(report):
    from pymrst.research_studio.eclipse import describe_gaps
    return "; ".join(describe_gaps(report))


def test_seqnum_is_reported_as_used_because_the_reader_depends_on_it():
    """It was classified 'read but unused', which was simply wrong.

    `read_restart` splits report steps on SEQNUM; a reader ignoring it merges a
    whole simulation into one step. Listing the keyword the restart reader most
    depends on as unused would send someone auditing coverage in the wrong
    direction.
    """
    from pymrst.research_studio.eclipse import classify_keyword, keyword_support
    assert classify_keyword("SEQNUM") == "used"
    assert "SEQNUM" not in keyword_support()["read_but_unused"]


def test_unimplemented_and_unknown_keywords_are_separated():
    """Two different situations. LGR is a gap this package knows it has; an
    unrecognised keyword is one it does not know about at all, and the second
    is the one that should worry a reader more.

    LGR is the example rather than NNC because NNC moved to its own
    classification once `read_nnc` existed - it is readable, just not folded
    into the case. Keeping it here would have asserted a gap that is no longer
    the right description.
    """
    import tempfile
    from pymrst.research_studio.eclipse import (
        NOT_IMPLEMENTED, UNRECOGNISED, describe_gaps, keyword_coverage,
    )
    from pymrst.research_studio.eclipse.records import write_records

    path = Path(tempfile.mkdtemp())/"GAPS.INIT"
    write_records(path, [
        ("PORO", "REAL", np.ones(3)),
        ("LGRHEADI", "INTE", np.array([1, 2], dtype=np.int64)),
        ("COARSEN", "INTE", np.ones(2, dtype=np.int64)),
        ("WEIRDKW", "REAL", np.ones(2)),
        ("INTEHEAD", "INTE", np.zeros(95, dtype=np.int64)),
    ])
    report = keyword_coverage(path)
    assert report.of("used") == ("PORO",)
    assert report.of(NOT_IMPLEMENTED) == ("COARSEN", "LGRHEADI")
    assert report.of(UNRECOGNISED) == ("WEIRDKW",)
    assert not report.complete

    lines = describe_gaps(report)
    assert any("local grid refinement" in line for line in lines), (
        "a named gap must say what is lost, not just that something is")
    assert any("WEIRDKW" in line and "used by nothing" in line for line in lines)


def test_metadata_alone_does_not_make_a_file_incomplete():
    import tempfile
    from pymrst.research_studio.eclipse import keyword_coverage
    from pymrst.research_studio.eclipse.records import write_records
    path = Path(tempfile.mkdtemp())/"META.INIT"
    write_records(path, [("PORO", "REAL", np.ones(3)),
                         ("INTEHEAD", "INTE", np.zeros(95, dtype=np.int64))])
    report = keyword_coverage(path)
    assert report.of("read_but_unused") == ("INTEHEAD",)
    assert report.complete


def test_marker_records_are_not_counted_as_dropped_keywords():
    """ENDGRID carries no data. Counting it would report a gap that is not one."""
    import tempfile
    from pymrst.research_studio.eclipse import keyword_coverage
    from pymrst.research_studio.eclipse.records import write_records
    path = Path(tempfile.mkdtemp())/"MARK.EGRID"
    write_records(path, [("COORD", "REAL", np.ones(6)), ("ENDGRID", "MESS", [])])
    report = keyword_coverage(path)
    assert "ENDGRID" not in report.classification
    assert report.complete


def test_repeated_keywords_are_counted_not_collapsed():
    """A UNRST repeats PRESSURE once per report step; a count of 1 would
    suggest a single-step file."""
    import tempfile
    from pymrst.research_studio.eclipse import keyword_coverage
    from pymrst.research_studio.eclipse.records import write_records
    path = Path(tempfile.mkdtemp())/"REP.UNRST"
    records = []
    for step in range(3):
        records.append(("SEQNUM", "INTE", np.array([step], dtype=np.int64)))
        records.append(("PRESSURE", "REAL", np.ones(4)))
    write_records(path, records)
    assert keyword_coverage(path).counts["PRESSURE"] == 3


def test_a_case_with_no_eclipse_output_is_refused():
    import tempfile
    from pymrst.research_studio.eclipse import case_coverage
    with pytest.raises(FileNotFoundError, match="no ECLIPSE output"):
        case_coverage(Path(tempfile.mkdtemp())/"ABSENT")


def test_missing_files_are_omitted_rather_than_reported_empty():
    import tempfile
    from pymrst.research_studio.eclipse import case_coverage, write_eclipse_case
    text, grid, _, _ = _norne_case()
    stem = Path(tempfile.mkdtemp())/"BARE"
    write_eclipse_case(stem, grid, grdecl=text)          # grid only
    assert set(case_coverage(stem)) == {"EGRID"}


# --------------------------------------------------------------------------
# Non-neighbour connections (M2.6 remainder)

def _nnc_grid_records(actnum):
    """EGRID records for a 2x2x2 box with the given ACTNUM."""
    coord = []
    for j in range(3):
        for i in range(3):
            coord += [float(i), float(j), 0.0, float(i), float(j), 10.0]
    # 8 corners per cell over 8 cells = 64, laid out as 2*nx by 2*ny per
    # (layer, surface) pair - so 16 values each, not 8.
    zcorn = [float(k + top) for k in range(2) for top in (0, 1)
             for _ in range(16)]
    return [("FILEHEAD", "INTE", np.zeros(100, dtype=np.int64)),
            ("GRIDHEAD", "INTE", np.array([1, 2, 2, 2] + [0]*96, dtype=np.int64)),
            ("COORD", "REAL", np.array(coord, dtype=float)),
            ("ZCORN", "REAL", np.array(zcorn, dtype=float)),
            ("ACTNUM", "INTE", np.asarray(actnum, dtype=np.int64))]


def _nnc_case(pairs, transmissibility=None, actnum=None, extra=()):
    """Write a case carrying `pairs` (zero-based ACTIVE indices)."""
    import tempfile
    from pymrst.research_studio.eclipse import write_nnc
    from pymrst.research_studio.eclipse.records import write_records
    if actnum is None:
        actnum = np.ones(8, dtype=np.int64)
    stem = Path(tempfile.mkdtemp())/"N"
    records = _nnc_grid_records(actnum) + list(extra)
    records += write_nnc(None, pairs, actnum=actnum)
    write_records(stem.with_suffix(".EGRID"), records)
    if transmissibility is not None:
        write_records(stem.with_suffix(".INIT"),
                      [("TRANNNC", "REAL", np.asarray(transmissibility, float))])
    return stem


def test_non_neighbour_connections_survive_the_global_to_active_mapping():
    """The trap, again, and the reason this needs inactive cells to test.

    ECLIPSE writes NNC1/NNC2 as one-based *global* indices. On a grid with
    every cell active that is indistinguishable from active numbering, so a
    reader that confuses them works perfectly on test models and joins the
    wrong cells on every real one.
    """
    from pymrst.research_studio.eclipse import read_nnc
    actnum = np.ones(8, dtype=np.int64)
    actnum[[1, 5]] = 0
    pairs = np.array([[0, 3], [1, 4]])            # zero-based active
    stem = _nnc_case(pairs, actnum=actnum)

    got = read_nnc(stem)
    assert len(got) == 2
    np.testing.assert_array_equal(got.cells, pairs)


def test_transmissibilities_are_paired_with_their_connections():
    from pymrst.research_studio.eclipse import read_nnc
    stem = _nnc_case(np.array([[0, 3], [1, 4]]), transmissibility=[1e-12, 2e-12])
    got = read_nnc(stem)
    np.testing.assert_allclose(got.transmissibility, [1e-12, 2e-12], rtol=1e-6)
    assert not got.sealed


def test_connections_that_all_transmit_nothing_are_reported_as_sealed():
    """Worth asking separately from "are there any". A case can list hundreds
    of NNCs that transmit nothing - it looks connected and behaves sealed."""
    from pymrst.research_studio.eclipse import read_nnc
    stem = _nnc_case(np.array([[0, 3], [1, 4]]), transmissibility=[0.0, 0.0])
    assert read_nnc(stem).sealed


def test_a_case_with_no_connections_reports_none_rather_than_failing():
    import tempfile
    from pymrst.research_studio.eclipse import read_nnc
    from pymrst.research_studio.eclipse.records import write_records
    stem = Path(tempfile.mkdtemp())/"NONE"
    write_records(stem.with_suffix(".EGRID"),
                  _nnc_grid_records(np.ones(8, dtype=np.int64)))
    got = read_nnc(stem)
    assert len(got) == 0
    assert "no NNC1/NNC2" in got.source["nnc"]


def test_unpaired_connection_ends_are_refused():
    import tempfile
    from pymrst.research_studio.eclipse import read_nnc
    from pymrst.research_studio.eclipse.records import write_records
    stem = Path(tempfile.mkdtemp())/"BAD"
    write_records(stem.with_suffix(".EGRID"),
                  _nnc_grid_records(np.ones(8, dtype=np.int64))
                  + [("NNC1", "INTE", np.array([1, 2, 3], dtype=np.int64)),
                     ("NNC2", "INTE", np.array([4, 5], dtype=np.int64))])
    with pytest.raises(ValueError, match="cannot be paired"):
        read_nnc(stem)


def test_a_connection_to_an_inactive_cell_is_refused_not_dropped():
    """Dropping it would quietly reduce the connectivity of the model."""
    import tempfile
    from pymrst.research_studio.eclipse import read_nnc
    from pymrst.research_studio.eclipse.records import write_records
    actnum = np.ones(8, dtype=np.int64)
    actnum[[1, 5]] = 0
    stem = Path(tempfile.mkdtemp())/"DEAD"
    write_records(stem.with_suffix(".EGRID"),
                  _nnc_grid_records(actnum)
                  + [("NNC1", "INTE", np.array([1], dtype=np.int64)),
                     ("NNC2", "INTE", np.array([2], dtype=np.int64))])  # global 2 is inactive
    with pytest.raises(ValueError, match="inactive"):
        read_nnc(stem)


def test_the_index_convention_is_chosen_not_guessed():
    """On an all-active grid the two conventions coincide, so a reader that
    detected automatically would be right here and wrong on real models."""
    from pymrst.research_studio.eclipse import read_nnc
    stem = _nnc_case(np.array([[0, 3]]))
    assert len(read_nnc(stem, indices="active")) == 1
    with pytest.raises(ValueError, match="'global' or 'active'"):
        read_nnc(stem, indices="whatever")


def test_connections_split_across_records_are_all_kept():
    """A large model splits NNC1 over several records; keeping the first would
    silently drop most of the connections while appearing to work."""
    import tempfile
    from pymrst.research_studio.eclipse import read_nnc
    from pymrst.research_studio.eclipse.records import write_records
    stem = Path(tempfile.mkdtemp())/"SPLIT"
    write_records(stem.with_suffix(".EGRID"),
                  _nnc_grid_records(np.ones(8, dtype=np.int64))
                  + [("NNC1", "INTE", np.array([1, 2], dtype=np.int64)),
                     ("NNC2", "INTE", np.array([3, 4], dtype=np.int64)),
                     ("NNC1", "INTE", np.array([5], dtype=np.int64)),
                     ("NNC2", "INTE", np.array([6], dtype=np.int64))])
    assert len(read_nnc(stem)) == 3


def test_mismatched_transmissibility_count_is_refused():
    from pymrst.research_studio.eclipse import read_nnc
    stem = _nnc_case(np.array([[0, 3], [1, 4]]), transmissibility=[1e-12])
    with pytest.raises(ValueError, match="refusing to pair them by position"):
        read_nnc(stem)


def test_coverage_calls_nnc_readable_but_absent_from_the_imported_case():
    """Neither overclaimed nor underclaimed.

    Calling NNC "used" would say the imported case contains those connections,
    which it does not; calling it "not implemented" would hide that the data is
    reachable through `read_nnc`. It gets its own classification, and the file
    is still reported incomplete because the *case* is sealed across them.
    """
    from pymrst.research_studio.eclipse import (
        AVAILABLE_SEPARATELY, describe_gaps, keyword_coverage,
    )
    stem = _nnc_case(np.array([[0, 3]]))
    report = keyword_coverage(stem.with_suffix(".EGRID"))
    assert set(report.of(AVAILABLE_SEPARATELY)) == {"NNC1", "NNC2"}
    assert not report.complete
    assert any("read_nnc" in line and "sealed across them" in line
               for line in describe_gaps(report))
