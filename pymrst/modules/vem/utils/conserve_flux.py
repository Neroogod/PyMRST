"""Make a VEM flux field locally conservative (MRST ``vem/utils/conserveFlux.m``).

A flux field that does not close the mass balance cell by cell is fatal for
transport: a cell whose fluxes do not balance manufactures or destroys
saturation at every step.

**Which PyMRST flux fields need this is worth knowing before reaching for it.**
Measured on a twisted 5x5 grid, `incomp_vem`'s default reconstruction and its
MPFA variant are already conservative to ~1e-15 - they recover a constant
`K grad p` per cell, whose flux through a closed surface cancels by
construction. The **TPFA** reconstruction is not: it is off by ~1e-1 on the
same problem, because a two-point transmissibility cannot represent the full
tensor on a non-orthogonal grid. That is the case this function is for, along
with any flux field arriving from outside the module.

The correction is a discrete Neumann problem. Find a cell potential `beta` with

    B beta = r,     r = q - div(flux)

where `B` is the graph Laplacian of the cell connectivity weighted by
`area / omega`, then correct each face by `area * (beta_left - beta_right) /
omega`. Because the correction is a difference of cell values, it cancels in
the sum over any closed surface - so it fixes the balance without introducing a
new imbalance elsewhere. Faces carrying a no-flow condition are excluded, since
their flux is prescribed and must not move.

`omega` weights how much of a cell's correction each of its faces absorbs. The
default weights by the normal permeability `n' K n`, so a correction flows
preferentially through the directions the rock actually conducts.

**The weights must be the same on both sides of a face**, or the correction
does not telescope and the result is not conservative. `permWeighted` and
`ones` are face-symmetric by construction. `tpf` is not: it is a half-face
transmissibility, and on a non-orthogonal grid the two halves differ - by 15%
on the twisted grid in the tests below - so it leaves a residual. MRST has the
same property and the same outcome, which is why it ends with a "could not
construct conservative flux field" warning rather than an assertion. That
warning is reproduced here; `tpf` is offered for parity, not as a working
choice on a distorted grid.

Returns the corrected state **and** the residual before correction, as MRST
does: the caller usually wants to know how far off the flux was, not only that
it has been fixed.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.params.rock.perm_tensor import perm_tensor
from pymrst.utils.gridtools.boundary_faces import boundary_faces
from pymrst.utils.rldecode import rldecode

WEIGHTS = ("permWeighted", "tpf", "ones")


def _face_signs(g) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-face list and +1/-1 for whether the face points out of its cell."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    faces = np.asarray(g.cells.faces, dtype=np.int64)
    faces = faces[:, 0] if faces.ndim > 1 else faces
    counts = np.diff(face_pos)
    owner = rldecode(np.arange(g.cells.num), counts)
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    signs = np.where(neighbours[faces, 0] == owner, 1.0, -1.0)
    return faces, signs


def _face_weights(g, rock, kind: str) -> np.ndarray:
    """Per-cell-face weights; see the module docstring for what they mean."""
    faces, signs = _face_signs(g)
    if kind == "ones":
        return np.ones(faces.size)
    if kind == "tpf":
        from pymrst.solvers.compute_trans import compute_trans
        return compute_trans(g, rock) / np.asarray(g.faces.areas)[faces]
    if kind != "permWeighted":
        raise ValueError(
            f"faceWeights must be one of {WEIGHTS}, not {kind!r}")

    dim = g.griddim
    # perm_tensor returns (values, i, j); the first is (ncells, dim^2).
    tensor = np.asarray(perm_tensor(rock, dim)[0],
                        dtype=float).reshape(-1, dim, dim)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    counts = np.diff(face_pos)
    owner = rldecode(np.arange(g.cells.num), counts)
    normals = (np.asarray(g.faces.normals, dtype=float)[faces]
               * (signs / np.asarray(g.faces.areas)[faces])[:, None])

    # delta = n' K n on each cell-face, with n the unit outward normal.
    delta = np.einsum("ij,ijk,ik->i", normals, tensor[owner], normals)

    # omega(face) = sum(delta) / (count * prod(delta)) over that face's
    # cell-faces: the mean of the reciprocals, so a face resists as the
    # harmonic combination of the two sides. A boundary face has one value and
    # reduces to 1.
    omega = np.zeros(g.faces.num)
    for face in range(g.faces.num):
        d = delta[faces == face]
        omega[face] = d.sum() / (d.size * np.prod(d))
    return omega[faces]


def conserve_flux(state, g, rock, bc=None, src=None,
                  face_weights: str = "permWeighted",
                  tol: float = 1e-14) -> Tuple[dict, np.ndarray]:
    """Correct ``state['flux']`` so each cell balances; return the residual."""
    faces, signs = _face_signs(g)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    counts = np.diff(face_pos)
    owner = rldecode(np.arange(g.cells.num), counts)
    areas = np.asarray(g.faces.areas, dtype=float)

    # No-flow faces: the outer boundary, less anything given a pressure.
    no_flow = np.zeros(g.faces.num, dtype=bool)
    no_flow[boundary_faces(g)] = True
    if bc is not None and getattr(bc, "face", None) is not None \
            and not hasattr(bc, "func"):
        for f, kind in zip(bc.face, bc.type):
            if kind == "pressure":
                no_flow[int(f)] = False

    rhs = np.zeros(g.cells.num)
    if src is not None:
        np.add.at(rhs, np.asarray(src.cell, dtype=np.int64),
                  np.asarray(src.rate, dtype=float))

    flux = np.asarray(state["flux"], dtype=float).copy()
    divergence = np.zeros(g.cells.num)
    np.add.at(divergence, owner, flux[faces] * signs)
    residual = rhs - divergence

    scale = rhs if np.any(rhs) else flux
    if np.linalg.norm(residual) / max(np.linalg.norm(scale), 1e-300) < tol:
        # MRST warns here. Already-conservative input is a legitimate state -
        # a TPFA flux, say - so it is not an error, but the caller asked for a
        # correction and is getting none.
        import warnings
        warnings.warn("flux is already conservative; no postprocessing needed")
        return state, residual

    omega = _face_weights(g, rock, face_weights)
    weight = areas[faces] / omega

    # Off-diagonal: -weight between the two cells sharing an interior face.
    other = np.where(np.asarray(g.faces.neighbors)[faces, 0] == owner,
                     np.asarray(g.faces.neighbors)[faces, 1],
                     np.asarray(g.faces.neighbors)[faces, 0])
    interior = other >= 0
    B = sp.coo_array((-weight[interior], (owner[interior], other[interior])),
                     shape=(g.cells.num, g.cells.num)).tocsr()
    # Diagonal: the same weights, but only over faces that may carry flux.
    movable = np.where(no_flow[faces], 0.0, 1.0)
    diagonal = np.zeros(g.cells.num)
    np.add.at(diagonal, owner, movable * weight)
    B = B + sp.diags(diagonal, format="csr")

    if no_flow.sum() == boundary_faces(g).size:
        # Every boundary face is no-flow, so beta is determined only up to a
        # constant; pin it exactly as the pressure system is pinned.
        B = B.tolil()
        B[0, 0] *= 2.0
        B = B.tocsr()

    beta = sp.linalg.spsolve(B.tocsc(), residual)
    beta_cf = beta[owner]
    beta_cf[no_flow[faces]] = 0.0

    correction = np.zeros(g.faces.num)
    np.add.at(correction, faces, movable * beta_cf / omega * signs)
    flux = flux + areas * correction

    divergence = np.zeros(g.cells.num)
    np.add.at(divergence, owner, flux[faces] * signs)
    remaining = rhs - divergence
    if np.linalg.norm(remaining) / max(np.linalg.norm(scale), 1e-300) > tol:
        import warnings
        warnings.warn("could not construct a conservative flux field")

    state = dict(state)
    state["flux"] = flux
    return state, residual


conserveFlux = conserve_flux


if __name__ == "__main__":
    import warnings

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc.add_bc import add_bc
    from pymrst.params.wells_and_bc.add_source import add_source
    from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
    from pymrst.modules.vem.incomp_vem import incomp_vem
    from pymrst.modules.vem.utils.compute_vem_geometry import (
        compute_vem_geometry,
    )

    def _twisted(nx=5, amp=0.3):
        """A non-orthogonal grid, where two-point transmissibility fails."""
        g = cart_grid([nx, nx], [float(nx), float(nx)])
        c = np.asarray(g.nodes.coords, dtype=float)
        x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
        c[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        c[:, 1] += 0.6 * amp * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
        g.nodes.coords = c
        return compute_vem_geometry(compute_geometry(g))

    def _imbalance(g, flux, rhs=None):
        faces, signs = _face_signs(g)
        owner = rldecode(np.arange(g.cells.num),
                         np.diff(np.asarray(g.cells.facePos, dtype=np.int64)))
        divergence = np.zeros(g.cells.num)
        np.add.at(divergence, owner, flux[faces] * signs)
        return (np.zeros(g.cells.num) if rhs is None else rhs) - divergence

    def _driven(trans="tpfa", nx=5):
        """A left-to-right pressure drop, solved and reconstructed with `trans`."""
        g = _twisted(nx)
        rock = make_rock(g, 1.0, 1.0)
        S = compute_virtual_ip(g, rock, 1, trans=trans)
        centroids = np.asarray(g.faces.centroids)
        outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
        left = np.flatnonzero(
            (np.abs(centroids[:, 0] - centroids[:, 0].min()) < 1e-9) & outer)
        right = np.flatnonzero(
            (np.abs(centroids[:, 0] - centroids[:, 0].max()) < 1e-9) & outer)
        bc = add_bc(None, left, "pressure", np.ones(left.size))
        bc = add_bc(bc, right, "pressure", np.zeros(right.size))
        return g, rock, bc, incomp_vem(g, S, bc=bc)

    def test_the_tpfa_flux_is_the_one_that_needs_correcting() -> None:
        """States the premise the rest of these tests rest on."""
        _, _, _, tpfa = _driven("tpfa")
        g, _, _, default = _driven(None)
        assert np.linalg.norm(_imbalance(g, tpfa["flux"])) > 1e-3
        assert np.linalg.norm(_imbalance(g, default["flux"])) < 1e-10
        print(f"ok test_the_tpfa_flux_is_the_one_that_needs_correcting "
              f"(tpfa {np.linalg.norm(_imbalance(g, tpfa['flux'])):.1e}, "
              f"default {np.linalg.norm(_imbalance(g, default['flux'])):.1e})")

    def test_the_correction_closes_the_mass_balance() -> None:
        g, rock, bc, state = _driven("tpfa")
        before = np.linalg.norm(_imbalance(g, state["flux"]))
        fixed, _ = conserve_flux(state, g, rock, bc=bc)
        after = np.linalg.norm(_imbalance(g, fixed["flux"]))
        assert after < 1e-10 and after < before, (before, after)
        print(f"ok test_the_correction_closes_the_mass_balance "
              f"({before:.2e} -> {after:.2e})")

    def test_the_returned_residual_is_the_one_before_correction() -> None:
        """MRST returns it so a caller can see how far off the flux was."""
        g, rock, bc, state = _driven("tpfa")
        before = _imbalance(g, state["flux"])
        _, residual = conserve_flux(state, g, rock, bc=bc)
        assert np.allclose(residual, before), np.abs(residual - before).max()
        print("ok test_the_returned_residual_is_the_one_before_correction")

    def test_the_input_state_is_not_modified() -> None:
        g, rock, bc, state = _driven("tpfa")
        original = state["flux"].copy()
        conserve_flux(state, g, rock, bc=bc)
        assert np.allclose(state["flux"], original), "input was mutated"
        print("ok test_the_input_state_is_not_modified")

    def test_sealed_faces_keep_their_flux() -> None:
        """The correction may not leak out through a no-flow boundary."""
        g, rock, bc, state = _driven("tpfa")
        fixed, _ = conserve_flux(state, g, rock, bc=bc)
        centroids = np.asarray(g.faces.centroids)
        outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
        driven = np.zeros(g.faces.num, dtype=bool)
        for f in np.concatenate([bc.face]):
            driven[int(f)] = True
        sealed = np.flatnonzero(outer & ~driven)
        assert np.allclose(fixed["flux"][sealed], state["flux"][sealed])
        print(f"ok test_sealed_faces_keep_their_flux ({sealed.size} faces)")

    def test_a_source_is_balanced_rather_than_zeroed() -> None:
        g = _twisted(5)
        rock = make_rock(g, 1.0, 1.0)
        S = compute_virtual_ip(g, rock, 1, trans="tpfa")
        centroids = np.asarray(g.faces.centroids)
        outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
        right = np.flatnonzero(
            (np.abs(centroids[:, 0] - centroids[:, 0].max()) < 1e-9) & outer)
        bc = add_bc(None, right, "pressure", np.zeros(right.size))
        src = add_source(None, [0], [1.0])
        state = incomp_vem(g, S, bc=bc, src=src)
        rhs = np.zeros(g.cells.num)
        rhs[0] = 1.0
        fixed, _ = conserve_flux(state, g, rock, bc=bc, src=src)
        after = np.linalg.norm(_imbalance(g, fixed["flux"], rhs))
        assert after < 1e-9, after
        print(f"ok test_a_source_is_balanced_rather_than_zeroed ({after:.1e})")

    def test_the_face_symmetric_weightings_converge() -> None:
        """`permWeighted` and `ones` give the same weight to both sides."""
        for kind in ("permWeighted", "ones"):
            g, rock, bc, state = _driven("tpfa")
            fixed, _ = conserve_flux(state, g, rock, bc=bc, face_weights=kind)
            after = np.linalg.norm(_imbalance(g, fixed["flux"]))
            assert after < 1e-9, (kind, after)
        print("ok test_the_face_symmetric_weightings_converge "
              "(permWeighted, ones)")

    def test_the_half_transmissibility_weighting_does_not_converge() -> None:
        """And says so, rather than returning a field that is quietly wrong.

        A half-face transmissibility differs between the two sides of a face
        on a non-orthogonal grid, so the correction does not telescope. MRST
        behaves the same way and issues the same warning.
        """
        g, rock, bc, state = _driven("tpfa")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fixed, _ = conserve_flux(state, g, rock, bc=bc, face_weights="tpf")
        after = np.linalg.norm(_imbalance(g, fixed["flux"]))
        assert after > 1e-9, after
        assert any("could not construct" in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
        print(f"ok test_the_half_transmissibility_weighting_does_not_converge "
              f"(residual {after:.1e}, warned)")

    def test_the_weights_are_symmetric_only_for_the_working_choices() -> None:
        """The mechanism behind the two tests above, measured directly."""
        import collections
        g, rock, _, _ = _driven("tpfa")
        faces, _ = _face_signs(g)
        asymmetry = {}
        for kind in WEIGHTS:
            weights = _face_weights(g, rock, kind)
            by_face = collections.defaultdict(list)
            for cf, f in enumerate(faces):
                by_face[int(f)].append(weights[cf])
            interior = [v for v in by_face.values() if len(v) == 2]
            asymmetry[kind] = max(abs(a - b) / max(abs(a), 1e-30)
                                  for a, b in interior)
        assert asymmetry["permWeighted"] == 0.0 and asymmetry["ones"] == 0.0
        assert asymmetry["tpf"] > 1e-3, asymmetry["tpf"]
        print(f"ok test_the_weights_are_symmetric_only_for_the_working_choices "
              f"(tpf asymmetry {asymmetry['tpf']:.1%})")

    def test_the_weighting_matters_when_the_rock_is_anisotropic() -> None:
        """With `K = I` the two coincide, which is why this uses a tensor.

        `permWeighted` reduces to `ones` for isotropic unit permeability -
        `n' K n` is then 1 on every face - so a homogeneous test cannot tell
        the two apart, and would report a weighting that does nothing as
        working.
        """
        g = _twisted(5)
        anisotropic = make_rock(
            g, np.array([[4.0, 0.5, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, anisotropic, 1, trans="tpfa")
        centroids = np.asarray(g.faces.centroids)
        outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
        left = np.flatnonzero(
            (np.abs(centroids[:, 0] - centroids[:, 0].min()) < 1e-9) & outer)
        right = np.flatnonzero(
            (np.abs(centroids[:, 0] - centroids[:, 0].max()) < 1e-9) & outer)
        bc = add_bc(None, left, "pressure", np.ones(left.size))
        bc = add_bc(bc, right, "pressure", np.zeros(right.size))
        state = incomp_vem(g, S, bc=bc)

        first = conserve_flux(state, g, anisotropic, bc=bc,
                              face_weights="permWeighted")[0]["flux"]
        second = conserve_flux(state, g, anisotropic, bc=bc,
                               face_weights="ones")[0]["flux"]
        assert not np.allclose(first, second), "the weighting had no effect"
        for field in (first, second):
            assert np.linalg.norm(_imbalance(g, field)) < 1e-9
        print("ok test_the_weighting_matters_when_the_rock_is_anisotropic")

    def test_the_weightings_coincide_for_isotropic_unit_rock() -> None:
        """Stated so the previous test's fixture is not mistaken for arbitrary."""
        g, rock, bc, state = _driven("tpfa")
        first = conserve_flux(state, g, rock, bc=bc,
                              face_weights="permWeighted")[0]["flux"]
        second = conserve_flux(state, g, rock, bc=bc,
                               face_weights="ones")[0]["flux"]
        assert np.allclose(first, second)
        print("ok test_the_weightings_coincide_for_isotropic_unit_rock")

    def test_an_unknown_weighting_is_rejected() -> None:
        g, rock, bc, state = _driven("tpfa")
        try:
            conserve_flux(state, g, rock, bc=bc, face_weights="harmonic")
        except ValueError as error:
            assert "faceWeights" in str(error)
            print("ok test_an_unknown_weighting_is_rejected")
            return
        raise AssertionError("an unknown weighting must be rejected")

    def test_an_already_conservative_field_warns_and_is_unchanged() -> None:
        g, rock, bc, state = _driven(None)          # the default reconstruction
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            same, _ = conserve_flux(state, g, rock, bc=bc)
        assert any("already conservative" in str(w.message) for w in caught)
        assert np.allclose(same["flux"], state["flux"])
        print("ok test_an_already_conservative_field_warns_and_is_unchanged")

    test_the_tpfa_flux_is_the_one_that_needs_correcting()
    test_the_correction_closes_the_mass_balance()
    test_the_returned_residual_is_the_one_before_correction()
    test_the_input_state_is_not_modified()
    test_sealed_faces_keep_their_flux()
    test_a_source_is_balanced_rather_than_zeroed()
    test_the_face_symmetric_weightings_converge()
    test_the_half_transmissibility_weighting_does_not_converge()
    test_the_weights_are_symmetric_only_for_the_working_choices()
    test_the_weighting_matters_when_the_rock_is_anisotropic()
    test_the_weightings_coincide_for_isotropic_unit_rock()
    test_an_unknown_weighting_is_rejected()
    test_an_already_conservative_field_warns_and_is_unchanged()
    print("\nAll tests passed!")
