"""Boundary conditions moved to the coarse grid (MRST ``coarsenBC.m``).

Fine boundary conditions become coarse ones, with the two types handled
differently because they mean different things:

* **flux** is extensive, so a coarse face carries the *sum* of the fluxes on
  its fine faces;
* **pressure** is intensive, so a coarse face carries a *sample*: the value on
  the first of its fine faces.

Mixing the two on one coarse face is refused, as it is in MRST. A coarse face
that is part pressure and part flux has no single well-posed condition, and
choosing one silently would change the problem being solved.

Three properties of MRST's implementation are preserved deliberately, and each
is a limitation rather than a design choice. They are stated here because
matching them was the point, and because none of them is visible from the
result.

**A coarse face is kept only if its first fine face carries a condition.**
MRST samples `fconn(connPos(1:end-1))` and drops everything else. So a coarse
face with conditions on all but its first fine face contributes nothing at all,
and its flux is lost rather than reported.

**The pressure value is a sample, not an average.** MRST's own comment says
this "could be improved by finding the actual center face". Where the fine
pressures vary along a coarse face, the coarse value is whichever one happened
to be first, which depends on face numbering.

**Flux is summed without the orientation sign.** `coarsenBC` computes
`fineToCoarseSign` and then overwrites the result with an unsigned sum; the
comment marks the block as a hack. `coarsen_flux` does apply the sign. The
difference only shows on a coarse face whose fine faces point both ways, which
a boundary coarse face on a Cartesian grid never does. `signed=True` applies
it; the default keeps MRST's answer.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign


def coarsen_bc(cg, bc, signed: bool = False):
    """Map fine boundary conditions onto the coarse grid.

    Returns a new `BoundaryCondition`, or ``None`` when nothing survives.
    """
    if bc is None or np.asarray(bc.face).size == 0:
        return bc
    if cg.parent is None:
        raise ValueError("the coarse grid has no parent; coarsening needs it")

    from pymrst.params.wells_and_bc.add_bc import add_bc

    n_fine = int(cg.parent.faces.num)
    conn_pos = np.asarray(cg.conn_pos, dtype=np.int64)
    fconn = np.asarray(cg.fconn, dtype=np.int64)
    counts = np.diff(conn_pos)
    face_no = np.repeat(np.arange(counts.size, dtype=np.int64), counts)

    # Fine face -> coarse face, but only for each coarse face's first fine face.
    representative = np.full(n_fine, -1, dtype=np.int64)
    representative[fconn[conn_pos[:-1]]] = np.arange(counts.size, dtype=np.int64)

    faces = np.asarray(bc.face, dtype=np.int64).ravel()
    values = np.asarray(bc.value, dtype=float).ravel()
    types = [str(t).lower() for t in bc.type]

    _refuse_mixed_conditions(faces, types, fconn, face_no, counts, n_fine)

    # Flux on a coarse face is the sum over all its fine faces, not just the
    # representative one.
    fine_flux = np.zeros(n_fine)
    is_flux = np.array([t == "flux" for t in types])
    fine_flux[faces[is_flux]] = values[is_flux]
    weight = fine_to_coarse_sign(cg) if signed else 1.0
    coarse_flux = np.bincount(face_no, weights=fine_flux[fconn] * weight,
                              minlength=counts.size)

    kept = representative[faces] >= 0
    if not np.any(kept):
        return None
    coarse_face = representative[faces[kept]]
    coarse_type = [types[i] for i in np.flatnonzero(kept)]
    coarse_value = values[kept].copy()
    on_flux = np.array([t == "flux" for t in coarse_type])
    coarse_value[on_flux] = coarse_flux[coarse_face[on_flux]]

    out = None
    for kind in ("pressure", "flux"):
        sel = np.array([t == kind for t in coarse_type])
        if np.any(sel):
            out = add_bc(out, coarse_face[sel], kind, coarse_value[sel])
    return out


def _refuse_mixed_conditions(faces, types, fconn, face_no, counts, n_fine):
    """A coarse face must be all pressure or no pressure."""
    is_pressure = np.zeros(n_fine, dtype=bool)
    for f, t in zip(faces, types):
        if t == "pressure":
            is_pressure[f] = True
    n_pressure = np.bincount(face_no, weights=is_pressure[fconn].astype(float),
                             minlength=counts.size)
    bad = np.flatnonzero((n_pressure > 0) & (n_pressure < counts))
    if bad.size:
        raise ValueError(
            f"coarse face {int(bad[0])} carries pressure on "
            f"{int(n_pressure[bad[0]])} of its {int(counts[bad[0]])} fine faces; "
            "mixed conditions have no well-posed coarse equivalent")


__all__ = ["coarsen_bc"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
    from pymrst.params.wells_and_bc.add_bc import add_bc

    def _case(dims=(4, 4, 1), part=(2, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims), list(part))
        return g, p, generate_coarse_grid(g, p)

    def _west_faces(g):
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        c = np.asarray(g.faces.centroids, dtype=float)
        return np.flatnonzero(np.any(nb < 0, axis=1) & (c[:, 0] < 1e-9))

    def test_flux_is_summed_over_the_coarse_face():
        """Extensive: the coarse face must carry what the fine faces carried.

        Sampling instead would give a smaller, entirely plausible number, and
        the model would quietly lose most of its injection.
        """
        g, _, cg = _case()
        west = _west_faces(g)
        assert west.size == 4, west
        bc = add_bc(None, west, "flux", np.full(west.size, 2.5))
        out = coarsen_bc(cg, bc)
        assert out is not None
        assert np.isclose(np.sum(out.value), 4 * 2.5), out.value
        for f, v in zip(out.face, out.value):
            fine = cg.sub_faces(int(f))
            assert np.isclose(v, 2.5 * np.count_nonzero(np.isin(west, fine))), f
        print(f"ok test_flux_is_summed_over_the_coarse_face "
              f"({out.face.size} coarse faces, total {out.value.sum()})")

    def test_pressure_is_sampled_not_summed():
        """Intensive: summing would multiply the pressure by the face count."""
        g, _, cg = _case()
        west = _west_faces(g)
        bc = add_bc(None, west, "pressure", np.full(west.size, 300.0))
        out = coarsen_bc(cg, bc)
        assert np.allclose(out.value, 300.0), out.value
        print("ok test_pressure_is_sampled_not_summed")

    def test_the_pressure_value_is_the_first_face_not_the_mean():
        """MRST samples, and its own comment calls that improvable.

        Recorded as a test so the behaviour is deliberate rather than
        accidental: the coarse value depends on face numbering.
        """
        g, _, cg = _case()
        west = _west_faces(g)
        values = np.array([100.0, 200.0, 300.0, 400.0])
        bc = add_bc(None, west, "pressure", values)
        out = coarsen_bc(cg, bc)
        for f, v in zip(out.face, out.value):
            first = int(cg.sub_faces(int(f))[0])
            expected = values[np.flatnonzero(west == first)[0]]
            assert np.isclose(v, expected), (f, v, expected)
            assert not np.isclose(v, values.mean()) or values.std() == 0
        print("ok test_the_pressure_value_is_the_first_face_not_the_mean")

    def test_mixed_conditions_are_refused():
        """No well-posed coarse condition exists, so neither choice is right."""
        g, _, cg = _case()
        west = _west_faces(g)
        # Put pressure on one fine face of a coarse face and flux on the rest.
        target = cg.sub_faces(int(np.flatnonzero(
            np.isin(cg.fconn[cg.conn_pos[:-1]], west))[0]))
        pressure_face = np.array([target[0]])
        flux_faces = np.setdiff1d(west, pressure_face)
        bc = add_bc(None, pressure_face, "pressure", 300.0)
        bc = add_bc(bc, flux_faces, "flux", 1.0)
        try:
            coarsen_bc(cg, bc)
        except ValueError as exc:
            assert "mixed conditions" in str(exc), exc
        else:
            raise AssertionError("expected mixed conditions to be refused")
        print("ok test_mixed_conditions_are_refused")

    def test_a_coarse_face_without_a_condition_on_its_first_fine_face_is_dropped():
        """MRST's real limitation, asserted so it stays visible.

        The flux on the other fine faces is not reported anywhere. A caller
        who expects conservation here will not get it.
        """
        g, _, cg = _case()
        west = _west_faces(g)
        firsts = cg.fconn[cg.conn_pos[:-1]]
        without_first = np.setdiff1d(west, firsts)
        assert without_first.size > 0, "the case cannot show the limitation"
        bc = add_bc(None, without_first, "flux", 1.0)
        assert coarsen_bc(cg, bc) is None, "a dropped face was reported anyway"
        print(f"ok test_a_coarse_face_without_a_condition_on_its_first_fine_face"
              f"_is_dropped ({without_first.size} faces silently lost)")

    def test_the_orientation_sign_is_off_by_default():
        """MRST's hack drops it; `signed=True` restores it.

        On a Cartesian boundary every fine face of a coarse face points the
        same way, so the two agree and the flag changes nothing here. The
        assertion records which default is MRST's.
        """
        g, _, cg = _case()
        west = _west_faces(g)
        bc = add_bc(None, west, "flux", 1.0)
        plain = coarsen_bc(cg, bc)
        signed = coarsen_bc(cg, bc, signed=True)
        assert np.allclose(np.abs(plain.value), np.abs(signed.value))
        print("ok test_the_orientation_sign_is_off_by_default")

    def test_both_types_survive_together_on_separate_faces():
        g, _, cg = _case()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        c = np.asarray(g.faces.centroids, dtype=float)
        boundary = np.any(nb < 0, axis=1)
        west = np.flatnonzero(boundary & (c[:, 0] < 1e-9))
        east = np.flatnonzero(boundary & (c[:, 0] > 4.0 - 1e-9))
        bc = add_bc(None, west, "flux", 1.0)
        bc = add_bc(bc, east, "pressure", 200.0)
        out = coarsen_bc(cg, bc)
        assert set(out.type) == {"flux", "pressure"}, out.type
        print(f"ok test_both_types_survive_together_on_separate_faces "
              f"({out.face.size} coarse conditions)")

    def test_empty_input_is_returned():
        _, _, cg = _case()
        assert coarsen_bc(cg, None) is None
        print("ok test_empty_input_is_returned")

    def test_a_grid_without_a_parent_is_refused():
        g, _, cg = _case()
        bc = add_bc(None, _west_faces(g), "flux", 1.0)
        cg.parent = None
        try:
            coarsen_bc(cg, bc)
        except ValueError as exc:
            assert "parent" in str(exc), exc
        else:
            raise AssertionError("expected a parentless coarse grid to fail")
        print("ok test_a_grid_without_a_parent_is_refused")

    test_flux_is_summed_over_the_coarse_face()
    test_pressure_is_sampled_not_summed()
    test_the_pressure_value_is_the_first_face_not_the_mean()
    test_mixed_conditions_are_refused()
    test_a_coarse_face_without_a_condition_on_its_first_fine_face_is_dropped()
    test_the_orientation_sign_is_off_by_default()
    test_both_types_survive_together_on_separate_faces()
    test_empty_input_is_returned()
    test_a_grid_without_a_parent_is_refused()
    print("\nAll tests passed!")
