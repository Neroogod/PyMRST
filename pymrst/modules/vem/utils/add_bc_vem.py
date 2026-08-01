"""Boundary conditions given as functions (MRST ``vem/utils/addBCVEM.m``).

`addBC` attaches a *value* to each face. This attaches a *function of position*
to a group of faces, which is what a VEM boundary condition usually wants: at
k=2 a face carries a moment as well as its nodes, so imposing an exact solution
means evaluating it at node coordinates, edge centroids and face centroids -
places the caller should not have to know about.

That is the whole reason this exists. Without it, prescribing a spatially
varying boundary condition means building a dof-index-to-value map by hand,
which forces the dof numbering out of `incomp_vem` and into every caller and
every test. `incomp_vem`'s `dirichlet_nodes` argument is still there and still
useful as a low-level entry point; it is no longer the only way in.

Each call adds one group, and groups may not overlap - a face with two
boundary conditions is a specification error, not a precedence question, so it
is rejected rather than resolved.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

TYPES = ("pressure", "flux")


@dataclass
class VemBC:
    """Groups of faces, each with a type and a function of position."""

    face: List[np.ndarray] = field(default_factory=list)
    type: List[str] = field(default_factory=list)
    func: List[Callable] = field(default_factory=list)
    sat: np.ndarray = None

    def __len__(self) -> int:
        return len(self.face)


def add_bc_vem(bc, faces, kind: str, func: Callable) -> VemBC:
    """Add a group of ``faces`` carrying ``func`` as a ``kind`` condition.

    ``func`` maps an ``(n, dim)`` array of points to ``n`` values.
    """
    faces = np.atleast_1d(np.asarray(faces, dtype=np.int64)).ravel()
    if faces.size == 0:
        # MRST warns and returns the struct unchanged. An empty face list is
        # usually a selection that matched nothing, so it is worth saying.
        import warnings
        warnings.warn("empty list of boundary faces", stacklevel=2)
        return bc if bc is not None else VemBC()

    if not isinstance(kind, str):
        raise TypeError("boundary condition type should be a string")
    kind = kind.lower()
    if kind not in TYPES:
        raise ValueError(
            f"boundary condition type should be 'pressure' or 'flux', "
            f"not {kind!r}")
    if not callable(func):
        raise TypeError("addBCVEM takes a function of position; use add_bc "
                        "for a constant value")

    bc = VemBC() if bc is None else bc
    for existing in bc.face:
        overlap = np.intersect1d(existing, faces)
        if overlap.size:
            raise ValueError(
                f"new boundary condition overlaps with conditions already "
                f"in the bc struct on face(s) {overlap[:5].tolist()}")

    bc.face.append(faces)
    bc.type.append(kind)
    bc.func.append(func)
    return bc


addBCVEM = add_bc_vem


if __name__ == "__main__":

    def test_a_group_is_recorded() -> None:
        bc = add_bc_vem(None, [1, 2, 3], "pressure", lambda P: P[:, 0])
        assert len(bc) == 1
        assert np.array_equal(bc.face[0], [1, 2, 3])
        assert bc.type[0] == "pressure"
        print("ok test_a_group_is_recorded")

    def test_several_groups_accumulate() -> None:
        bc = add_bc_vem(None, [1, 2], "pressure", lambda P: P[:, 0])
        bc = add_bc_vem(bc, [7, 8], "flux", lambda P: np.ones(len(P)))
        assert len(bc) == 2 and bc.type == ["pressure", "flux"]
        print("ok test_several_groups_accumulate")

    def test_the_type_is_case_insensitive() -> None:
        bc = add_bc_vem(None, [1], "PRESSURE", lambda P: P[:, 0])
        assert bc.type[0] == "pressure"
        print("ok test_the_type_is_case_insensitive")

    def test_overlapping_groups_are_rejected() -> None:
        """Two conditions on one face is a specification error, not a priority."""
        bc = add_bc_vem(None, [1, 2, 3], "pressure", lambda P: P[:, 0])
        try:
            add_bc_vem(bc, [3, 4], "flux", lambda P: np.ones(len(P)))
        except ValueError as error:
            assert "overlaps" in str(error) and "3" in str(error)
            print("ok test_overlapping_groups_are_rejected")
            return
        raise AssertionError("an overlapping group must be rejected")

    def test_disjoint_groups_are_allowed() -> None:
        bc = add_bc_vem(None, [1, 2], "pressure", lambda P: P[:, 0])
        bc = add_bc_vem(bc, [3, 4], "pressure", lambda P: P[:, 1])
        assert len(bc) == 2
        print("ok test_disjoint_groups_are_allowed")

    def test_an_unknown_type_is_rejected() -> None:
        try:
            add_bc_vem(None, [1], "dirichlet", lambda P: P[:, 0])
        except ValueError as error:
            assert "'pressure' or 'flux'" in str(error)
            print("ok test_an_unknown_type_is_rejected")
            return
        raise AssertionError("only pressure and flux exist")

    def test_a_constant_instead_of_a_function_is_rejected() -> None:
        """The likely mistake is reaching for add_bc's calling convention."""
        try:
            add_bc_vem(None, [1], "pressure", 42.0)
        except TypeError as error:
            assert "add_bc" in str(error)
            print("ok test_a_constant_instead_of_a_function_is_rejected")
            return
        raise AssertionError("a constant must be rejected")

    def test_an_empty_face_list_warns_and_changes_nothing() -> None:
        import warnings
        bc = add_bc_vem(None, [1, 2], "pressure", lambda P: P[:, 0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            same = add_bc_vem(bc, [], "flux", lambda P: np.ones(len(P)))
        assert len(same) == 1 and caught
        print("ok test_an_empty_face_list_warns_and_changes_nothing")

    def test_a_scalar_face_is_accepted() -> None:
        bc = add_bc_vem(None, 5, "flux", lambda P: np.ones(len(P)))
        assert np.array_equal(bc.face[0], [5])
        print("ok test_a_scalar_face_is_accepted")

    test_a_group_is_recorded()
    test_several_groups_accumulate()
    test_the_type_is_case_insensitive()
    test_overlapping_groups_are_rejected()
    test_disjoint_groups_are_allowed()
    test_an_unknown_type_is_rejected()
    test_a_constant_instead_of_a_function_is_rejected()
    test_an_empty_face_list_warns_and_changes_nothing()
    test_a_scalar_face_is_accepted()
    print("\nAll tests passed!")
