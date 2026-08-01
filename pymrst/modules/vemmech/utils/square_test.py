"""Square elasticity test cases for vemmech (MRST `squareTest`).

Builds a square (2-D) or cubic (3-D) domain, the boundary-condition sets for
its sides, and a list of ready-to-solve test cases. Two families come out:

- the *patch tests*, one per analytic field from `linear_displacement`, which
  a correct linear-elastic discretization must reproduce exactly;
- the *named cases* - hanging rods, cantilevers, loaded plates - which have no
  closed form and exist to be compared against MRST.

Each case carries everything a solve needs, so running one is
`vem_lin_elast(G, case.C, case.el_bc, case.load)`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.compute_geometry import compute_geometry
from pymrst.gridprocessing.triangulate_faces import triangulate_faces
from pymrst.modules.vemmech.utils.linear_displacement import linear_displacement
from pymrst.modules.vemmech.utils.square_grid import square_grid
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c, enu2c_2d
from pymrst.params.wells_and_bc.pside import pside
from pymrst.utils.mcolon import mcolon

SIDES_2D = ("Left", "Right", "Back", "Front")
SIDES_3D = ("Left", "Right", "Back", "Front", "Bottom", "Top")

# Every named case MRST defines, in its order. Note that MRST's `'all'`
# selects only the first block, not all of these - see `_cases_2d`.
TEST_NAMES_2D = (
    "original_2D_test",
    "hanging_rod_2D",
    "hanging_cantilever_2D",
    "weighted_slop_rod_2D_dir",
    "weighted_slop_rod_2D",
    "weighted_slop_rod_2D_nobcleftright",
    "rod_2D_pressure",
    "rod_2D_pressure_nobcleftright",
    "rod_2D_point_pressure",
    "rod_2D_point_pressure_noleftrighbc",
)
TEST_NAMES_3D = ("original_3D_test", "hanging_plate_3D")


@dataclass
class SideBC:
    """One side of the domain: its faces, and the nodes they pin."""

    face: np.ndarray
    nodes: np.ndarray
    mask: np.ndarray


@dataclass
class TestCase:
    """A solvable elasticity case: boundary data, material, load, expectation."""

    name: str
    el_bc: ElasticBC
    C: np.ndarray
    load: Callable[[np.ndarray], np.ndarray]
    disp_sol: Optional[Callable[[np.ndarray], np.ndarray]] = None
    solution: Any = None
    force_bc: Any = None


def square_test(
    *,
    L: Sequence[float] = (1.0, 1.0),
    cart_dims: Sequence[int] = (20, 20),
    grid_type: str = "cartgrid",
    disturb: float = 0.0,
    E: float = 1.0,
    nu: float = 0.3,
    make_testcases: bool = True,
    test_name: str = "all",
    triangulate: bool = True,
    every_case: bool = False,
) -> Tuple[Any, List[SideBC], List[TestCase]]:
    """Return ``(grid, side_bcs, test_cases)`` for the square test domain.

    ``test_name`` follows MRST: it selects one `switch` block, and ``'all'``
    selects the first of them rather than every one. ``every_case=True`` is a
    PyMRST convenience that builds all the named cases at once.
    """
    L = [float(v) for v in np.atleast_1d(L).ravel()]
    cart_dims = [int(n) for n in np.atleast_1d(cart_dims).ravel()]
    if len(cart_dims) != 2 and grid_type != "cartgrid":
        raise ValueError("only 'cartgrid' is defined outside two dimensions")

    grid = square_grid(cart_dims, L, grid_type=grid_type, disturb=disturb)
    if triangulate:
        grid = triangulate_faces(grid, np.arange(grid.faces.num))
    grid = create_augmented_grid(compute_geometry(grid))

    side_bcs = _side_boundary_conditions(grid, grid_type, L)
    build = enu2c if grid.griddim == 3 else enu2c_2d
    C = build(np.full(grid.cells.num, float(E)), np.full(grid.cells.num, float(nu)))

    cases: List[TestCase] = []
    if make_testcases:
        cases += _patch_test_cases(grid, side_bcs, C)
    selector = "__every__" if every_case else test_name
    cases += (_cases_3d(grid, side_bcs, C) if grid.griddim == 3
              else _cases_2d(grid, side_bcs, C, L, selector))
    return grid, side_bcs, cases


def _side_boundary_conditions(grid, grid_type: str, L) -> List[SideBC]:
    """One `SideBC` per side, in MRST's order."""
    dim = grid.griddim
    if grid_type == "cartgrid":
        sides = SIDES_3D if dim == 3 else SIDES_2D
        face_sets = [np.asarray(pside(None, grid, side, 0.0).face, dtype=np.int64)
                     for side in sides]
    else:
        if dim != 2:
            raise ValueError("non-Cartesian square tests are 2D only")
        centroids = np.asarray(grid.faces.centroids, dtype=float)
        tol = np.sqrt(np.finfo(float).eps)
        face_sets = [
            np.flatnonzero(np.abs(centroids[:, 0] - 0.0) < tol),
            np.flatnonzero(np.abs(centroids[:, 0] - L[0]) < tol),
            np.flatnonzero(np.abs(centroids[:, 1] - 0.0) < tol),
            np.flatnonzero(np.abs(centroids[:, 1] - L[1]) < tol),
        ]

    out = []
    for faces in face_sets:
        nodes = np.unique(
            np.asarray(grid.faces.nodes, dtype=np.int64)[
                mcolon(grid.faces.nodePos[faces], grid.faces.nodePos[faces + 1])
            ]
        ) if faces.size else np.zeros(0, dtype=np.int64)
        out.append(SideBC(face=faces, nodes=nodes,
                          mask=np.ones((nodes.size, dim), dtype=bool)))
    return out


def _dirichlet_case(grid, sides: Sequence[SideBC], C, load, bcdisp,
                    name: str, solution=None, force_bc=None) -> TestCase:
    """MRST's `makeDirTestCase`: stack the sides into one displacement BC."""
    dim = grid.griddim
    if sides:
        nodes = np.concatenate([s.nodes for s in sides])
        mask = np.vstack([s.mask for s in sides])
    else:
        nodes = np.zeros(0, dtype=np.int64)
        mask = np.zeros((0, dim), dtype=bool)
    values = (bcdisp(grid.nodes.coords[nodes, :dim]) if nodes.size
              else np.zeros((0, dim)))

    faces = (np.asarray(force_bc["faces"], dtype=np.int64) if force_bc
             else np.zeros(0, dtype=np.int64))
    force = (np.asarray(force_bc["force"], dtype=float) if force_bc
             else np.zeros((0, dim)))
    el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask, disp_vals=values,
                      force_faces=faces, force=force)
    return TestCase(name=name, el_bc=el_bc, C=C, load=load, disp_sol=bcdisp,
                    solution=solution, force_bc=force_bc)


def _patch_test_cases(grid, sides, C) -> List[TestCase]:
    """One case per analytic linear field, pinned on every side."""
    fields, names = linear_displacement(grid.griddim)
    zero_load = lambda x: np.zeros_like(x)                # noqa: E731
    return [
        _dirichlet_case(grid, sides, C, zero_load, field, name, solution=field)
        for field, name in zip(fields, names)
    ]


def _masked(side: SideBC, column: int) -> SideBC:
    """A copy of `side` free to move along `column`."""
    mask = side.mask.copy()
    mask[:, column] = False
    return SideBC(face=side.face, nodes=side.nodes, mask=mask)


def _cases_2d(grid, sides, C, L, test_name: str) -> List[TestCase]:
    """MRST's named 2-D cases, with MATLAB's `switch` semantics.

    MATLAB does not fall through: the first `case` block whose label list
    contains `test_name` runs, and no other. So `test_name='all'` yields only
    the *first* block, not every case - `all` selects a default rather than
    the whole set. Reproducing that matters, because a caller asking for
    `'all'` in MRST gets one case back.

    One block builds two cases, and only its first label selects it, exactly
    as in MRST. `every_case=True` on `square_test` steps outside MRST to build
    the full set, which is useful for testing but is not MRST behaviour.
    """
    Lx, Ly = L[0], L[1]
    zero = lambda x: np.zeros_like(x)                     # noqa: E731
    down = lambda x: np.tile([0.0, 1.0], (np.asarray(x).shape[0], 1))   # noqa: E731
    up = lambda x: -np.tile([0.0, 1.0], (np.asarray(x).shape[0], 1))    # noqa: E731
    none = lambda x: np.zeros((np.asarray(x).shape[0], 2))              # noqa: E731

    def face_force(faces, magnitude) -> dict:
        centroids = np.asarray(grid.faces.centroids, dtype=float)[faces]
        return {"faces": faces,
                "force": magnitude(centroids)[:, None] * np.array([0.0, -1.0])}

    def sliding_sides():
        return [_masked(sides[0], 1), _masked(sides[1], 1), _masked(sides[2], 0)]

    uniform = lambda cc: np.ones(cc.shape[0])             # noqa: E731
    sigma = Ly / 10.0
    bump = lambda amplitude: (                            # noqa: E731
        lambda cc: amplitude * np.exp(-(((cc[:, 0] - Lx / 2.0)) / sigma) ** 2)
    )

    def block_original():
        return [_dirichlet_case(grid, sides, C, down, zero, "original_2D_test")]

    def block_hanging():
        tenth = lambda x: down(x) * 0.1                   # noqa: E731
        return [
            _dirichlet_case(grid, sides[:2], C, down, zero, "hanging_rod_2D"),
            _dirichlet_case(grid, sides[:1], C, tenth, zero,
                            "hanging_cantilever_2D"),
        ]

    def block_slop_dir():
        return [_dirichlet_case(grid, sides[:3], C, up, zero,
                                "weighted_slop_rod_2D_dir")]

    def block_slop():
        return [_dirichlet_case(grid, sliding_sides(), C, up, zero,
                                "weighted_slop_rod_2D")]

    def block_slop_nolr():
        return [_dirichlet_case(grid, [_masked(sides[2], 0)], C, up, zero,
                                "weighted_slop_rod_2D_nobcleftright")]

    def block_pressure():
        return [_dirichlet_case(grid, sliding_sides(), C, none, zero,
                                "rod_2D_pressure",
                                force_bc=face_force(sides[3].face, uniform))]

    def block_pressure_nolr():
        return [_dirichlet_case(grid, [_masked(sides[2], 0)], C, none, zero,
                                "rod_2D_pressure_nobcleftright",
                                force_bc=face_force(sides[3].face, uniform))]

    def block_point():
        return [_dirichlet_case(grid, sliding_sides(), C, none, zero,
                                "rod_2D_point_pressure",
                                force_bc=face_force(sides[3].face, bump(1e-2)))]

    def block_point_nolr():
        return [_dirichlet_case(grid, [_masked(sides[2], 0)], C, none, zero,
                                "rod_2D_point_pressure_noleftrighbc",
                                force_bc=face_force(sides[3].face, bump(1.0)))]

    blocks = [
        (("original_2D_test", "all"), block_original),
        (("hanging_rod_2D", "all"), block_hanging),
        (("weighted_slop_rod_2D_dir", "all"), block_slop_dir),
        (("weighted_slop_rod_2D", "all"), block_slop),
        (("weighted_slop_rod_2D_nobcleftright", "all"), block_slop_nolr),
        (("rod_2D_pressure", "all"), block_pressure),
        (("rod_2D_pressure_nobcleftright", "all"), block_pressure_nolr),
        (("rod_2D_point_pressure", "all"), block_point),
        (("rod_2D_point_pressure_noleftrighbc", "all"), block_point_nolr),
    ]

    if test_name == "__every__":                          # not an MRST option
        return [case for _, build in blocks for case in build()]
    for labels, build in blocks:
        if test_name in labels:
            return build()
    raise ValueError(f"test name not recognized: {test_name!r}")


def _cases_3d(grid, sides, C) -> List[TestCase]:
    zero = lambda x: np.zeros_like(x)                     # noqa: E731
    lift = lambda x: np.tile([0.0, 0.0, 1.0], (np.asarray(x).shape[0], 1))  # noqa: E731
    return [
        _dirichlet_case(grid, sides, C, lift, zero, "original_3D_test"),
        _dirichlet_case(grid, sides[:4], C, lift, zero, "hanging_plate_3D"),
    ]


squareTest = square_test


if __name__ == "__main__":
    from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

    def test_cartesian_2d_builds_grid_sides_and_cases() -> None:
        grid, sides, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0])
        assert len(sides) == 4
        assert all(s.face.size > 0 for s in sides)
        names = [c.name for c in cases]
        assert names[:6] == ["cons_disp_1", "cons_disp_2", "lin_disp_1",
                             "lin_disp_2", "steatch_xy", "rot_xy"], names[:6]
        # MRST's 'all' selects one block, so only the first named case appears
        assert names[6:] == ["original_2D_test"], names[6:]
        print(f"ok test_cartesian_2d_builds_grid_sides_and_cases "
              f"({grid.cells.num} cells, {len(cases)} cases)")

    def test_3d_has_six_sides_and_its_two_cases() -> None:
        grid, sides, cases = square_test(
            cart_dims=[3, 3, 3], L=[1.0, 1.0, 1.0], triangulate=False)
        assert len(sides) == 6
        names = [c.name for c in cases]
        assert set(TEST_NAMES_3D) <= set(names), names
        print(f"ok test_3d_has_six_sides_and_its_two_cases ({len(cases)} cases)")

    def test_patch_cases_solve_to_their_analytic_field() -> None:
        """The reason the patch cases exist: they must be reproduced exactly."""
        grid, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                     triangulate=False, E=1e6, nu=0.25)
        checked = 0
        for case in cases:
            if case.solution is None:
                continue
            u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
            expected = case.solution(grid.nodes.coords[:, :2])
            error = np.abs(u - expected).max() / max(np.abs(expected).max(), 1.0)
            assert error < 1e-9, f"{case.name}: {error:.3e}"
            checked += 1
        print(f"ok test_patch_cases_solve_to_their_analytic_field ({checked} fields)")

    def test_named_cases_solve_and_respond_to_their_load() -> None:
        grid, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                     triangulate=False, E=1e6, nu=0.25,
                                     make_testcases=False, every_case=True)
        assert {c.name for c in cases} == set(TEST_NAMES_2D)
        for case in cases:
            u, _ = vem_lin_elast(grid, case.C, case.el_bc, case.load)
            assert np.all(np.isfinite(u)), case.name
            assert np.abs(u).max() > 0, f"{case.name} produced no displacement"
        print(f"ok test_named_cases_solve_and_respond_to_their_load "
              f"({len(cases)} cases)")

    def test_masked_sides_leave_one_direction_free() -> None:
        _, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                  triangulate=False, make_testcases=False,
                                  test_name="weighted_slop_rod_2D")
        mask = cases[0].el_bc.disp_mask
        assert mask[:, 0].any() and mask[:, 1].any()
        assert not mask.all(), "some component must be free to slide"
        print("ok test_masked_sides_leave_one_direction_free")

    def test_pressure_cases_carry_a_force_boundary() -> None:
        _, _, cases = square_test(cart_dims=[4, 4], L=[1.0, 1.0],
                                  triangulate=False, make_testcases=False,
                                  test_name="rod_2D_pressure")
        case = cases[0]
        assert case.el_bc.force_faces.size > 0
        assert np.allclose(case.el_bc.force[:, 0], 0.0)
        assert np.all(case.el_bc.force[:, 1] < 0), "load must push downwards"
        print("ok test_pressure_cases_carry_a_force_boundary")

    def test_unknown_test_name_is_rejected() -> None:
        try:
            square_test(cart_dims=[3, 3], test_name="nope", triangulate=False)
        except ValueError:
            print("ok test_unknown_test_name_is_rejected")
            return
        raise AssertionError("unknown test name should be rejected")

    test_cartesian_2d_builds_grid_sides_and_cases()
    test_3d_has_six_sides_and_its_two_cases()
    test_patch_cases_solve_to_their_analytic_field()
    test_named_cases_solve_and_respond_to_their_load()
    test_masked_sides_leave_one_direction_free()
    test_pressure_cases_carry_a_force_boundary()
    test_unknown_test_name_is_rejected()
    print("\nAll tests passed!")
