"""Coarse-grid generation (MRST ``generateCoarseGrid.m``).

A coarse grid is a topology, not a geometry: blocks of fine cells, and coarse
faces each standing for the set of fine faces separating one pair of blocks.
MRST records that as `cells.facePos`/`cells.faces`, `faces.neighbors`, and
`faces.connPos`/`faces.fconn` listing the fine faces behind each coarse face.
This builds the same structure for **any** partition.

A coarse face is not simply a block pair. MRST groups fine faces by the pair
*and* by an indicator, and the default indicator already carries the half-face
tag of every exterior face. So the boundary of a block is not one coarse face
but one per direction it faces: on a 6x4x2 grid in 3x2x1 blocks MRST reports 29
coarse faces, not the 13 that block pairs alone would give. `face_partition`
adds a further column, which is how a block pair is split into several coarse
faces.

`.g` is an addition rather than a port. When the partition happens to be a
rectangular Cartesian one, a tensor grid with the right block extents is built
and stored there, which is what several upscaling routines use to get block
geometry cheaply. For any other partition, including one that
`process_partition` has split across a fault, `.g` is ``None`` while the
topology is still complete. Earlier this function raised in that case, so a
coarse grid could not be built for a faulted model at all.
"""

from dataclasses import dataclass, field

import numpy as np

from pymrst.gridprocessing.geometry import compute_geometry
from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.modules.coarsegrid.cell_partition_to_face_partition import (
    cell_partition_to_face_partition,
)


@dataclass
class CoarseGrid:
    """A coarse grid and its mapping back to the fine model.

    `g` holds a geometric grid when the partition is rectangular Cartesian and
    is ``None`` otherwise. The topology fields below are always populated.
    """

    g: Grid | None
    partition: np.ndarray
    n_blocks: int
    parent: Grid | None = None
    # Coarse topology, mirroring MRST's coarse grid.
    neighbors: np.ndarray | None = None      # (n_coarse_faces, 2), -1 outside
    conn_pos: np.ndarray | None = None       # (n_coarse_faces + 1,) into fconn
    fconn: np.ndarray | None = None          # fine faces behind each coarse face
    face_pos: np.ndarray | None = None       # (n_blocks + 1,) into faces
    faces: np.ndarray | None = None          # coarse faces of each block
    face_indicator: np.ndarray | None = None  # (n_coarse_faces, k), MRST's `ind`
    griddim: int = 3
    # Geometry, filled in by `coarsen_geometry`. MRST keeps these on
    # `cg.cells`/`cg.faces`; the flat names here avoid shadowing `.g`.
    cell_volumes: np.ndarray | None = None
    cell_centroids: np.ndarray | None = None
    face_areas: np.ndarray | None = None
    face_centroids: np.ndarray | None = None
    face_normals: np.ndarray | None = None

    @property
    def num_faces(self) -> int:
        return 0 if self.neighbors is None else int(self.neighbors.shape[0])

    def blocks(self):
        order = np.argsort(self.partition, kind="stable")
        counts = np.bincount(self.partition, minlength=self.n_blocks)
        out, start = [], 0
        for c in counts:
            out.append(order[start:start + c])
            start += c
        return out

    def sub_faces(self, face: int) -> np.ndarray:
        """Fine faces making up one coarse face (MRST ``subFaces``)."""
        if self.fconn is None or self.conn_pos is None:
            raise ValueError("this coarse grid carries no face connectivity")
        return self.fconn[self.conn_pos[face]:self.conn_pos[face + 1]]

    def to_grid(self) -> Grid:
        """The coarse grid as a `Grid`, so ordinary solvers accept it.

        MRST states that a coarse grid "will in most aspects behave as the
        standard grids and can be used in many of the solvers in MRST", because
        its topology lives in the same `cells.facePos`/`faces.neighbors` fields
        a fine grid uses. This builds that view, so `compute_trans` and
        `incomp_tpfa` work on a coarse grid with no special casing.

        **It has no nodes**, and neither does MRST's. A coarse block is defined
        by which fine cells it holds, not by a polygon, so there is no vertex
        list to give. Everything that works from centroids, areas and normals
        is fine; anything needing vertices must go through `parent` and
        `sub_faces`, which is how the plotting routines already do it.

        Geometry is filled in by `coarsen_geometry` if it is not already there.
        """
        from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry

        if self.neighbors is None or self.face_pos is None:
            raise ValueError("this coarse grid carries no topology")
        if self.cell_volumes is None or self.face_areas is None:
            coarsen_geometry(self)

        n_faces = int(self.neighbors.shape[0])
        cells = Cells(
            num=int(self.n_blocks),
            facePos=np.asarray(self.face_pos, dtype=np.int64),
            faces=np.asarray(self.faces, dtype=np.int64),
            indexMap=np.arange(self.n_blocks, dtype=np.int64),
            volumes=self.cell_volumes,
            centroids=self.cell_centroids,
        )
        faces = Faces(
            num=n_faces,
            nodePos=np.zeros(n_faces + 1, dtype=np.int64),
            nodes=np.empty(0, dtype=np.int64),
            neighbors=np.asarray(self.neighbors, dtype=np.int64),
            areas=self.face_areas,
            normals=self.face_normals,
            centroids=self.face_centroids,
        )
        g = Grid(cells=cells, faces=faces,
                 nodes=Nodes(num=0, coords=np.empty((0, self.griddim))),
                 griddim=int(self.griddim), type=["generateCoarseGrid"])
        g.partition = self.partition
        g.parent = self.parent
        return g


def _default_indicator(g) -> np.ndarray:
    """MRST's implicit indicator: the half-face tag of every exterior face.

    Interior faces get 0. PyMRST numbers half-face tags from 0 and MRST from 1,
    so the tag is shifted by one; without that, tag 0 (West) would be
    indistinguishable from "interior" and every block's west boundary would
    merge with its interior connections.

    Returns a zero column when the grid carries no tags, which is what MRST
    does for a grid whose ``cells.faces`` has no tag column.
    """
    out = np.zeros((int(g.faces.num), 1), dtype=np.int64)
    tags = getattr(g.cells, "tags", None)
    if tags is None:
        return out
    tags = np.asarray(tags, dtype=np.int64).ravel()
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    if tags.size != cell_faces.size:
        return out
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    exterior = np.any(neighbors < 0, axis=1)
    on_boundary = exterior[cell_faces]
    out[cell_faces[on_boundary], 0] = tags[on_boundary] + 1
    return out


def _coarse_topology(g, partition: np.ndarray, n_blocks: int,
                     face_partition: np.ndarray | None = None):
    """Coarse faces, their fine faces, and the block-to-face mapping.

    A coarse face is a maximal set of fine faces sharing a block pair *and* an
    indicator value. Faces interior to a block are not coarse faces: they vanish
    into the block. Follows MRST's `coarseConnections` and
    `transposeConnections`, including their ordering, so a coarse face here
    carries the same number as in MATLAB.
    """
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    blocks = np.where(neighbors >= 0, partition[np.maximum(neighbors, 0)], -1)
    interior_to_a_block = (blocks[:, 0] == blocks[:, 1]) & np.all(neighbors >= 0, axis=1)

    keep = np.flatnonzero(~interior_to_a_block)
    indicator = _default_indicator(g)
    if face_partition is not None:
        extra = np.asarray(face_partition, dtype=np.int64).reshape(int(g.faces.num), -1)
        indicator = np.concatenate([indicator, extra], axis=1)

    # MRST sorts on [block pair, indicator, fine face] and run-length encodes
    # everything but the fine face. `np.unique` over the leading columns is the
    # same grouping, and it returns them in the same lexicographic order.
    key = np.concatenate([np.sort(blocks[keep], axis=1), indicator[keep]], axis=1)
    unique_key, group = np.unique(key, axis=0, return_inverse=True)
    group = group.astype(np.int64).ravel()

    order = np.lexsort((keep, group))
    counts = np.bincount(group, minlength=unique_key.shape[0])
    conn_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    fconn = keep[order].astype(np.int64)

    coarse_neighbors = unique_key[:, :2].astype(np.int64)
    face_indicator = unique_key[:, 2:].astype(np.int64)

    # Transpose: which coarse faces bound each block (MRST transposeConnections,
    # which sorts on [block, coarse face] so a block's faces come out ascending).
    n_faces = coarse_neighbors.shape[0]
    owners = coarse_neighbors.ravel(order="F")
    owned_face = np.tile(np.arange(n_faces, dtype=np.int64), 2)
    real = owners >= 0
    owners, owned_face = owners[real], owned_face[real]
    keep_order = np.lexsort((owned_face, owners))
    owners, owned_face = owners[keep_order], owned_face[keep_order]
    per_block = np.bincount(owners, minlength=n_blocks)
    face_pos = np.concatenate([[0], np.cumsum(per_block)]).astype(np.int64)
    return (coarse_neighbors, conn_pos, fconn, face_pos,
            owned_face.astype(np.int64), face_indicator)


def _tensor_coarse_grid(g, partition: np.ndarray, n_blocks: int):
    """A geometric coarse grid, when the partition is rectangular Cartesian.

    Returns ``None`` when the partition does not form a tensor product, which
    is the normal case for anything but `partition_cart_grid`.

    Logical indices come from `indexMap` rather than from cell order, because on
    a grid with cells removed the two disagree and the cell-order version reads
    the partition at the wrong logical positions. A logical slice with no cells
    left in it also means the grid no longer fills its box, so there is no
    tensor coarse grid to build and ``None`` is the answer.
    """
    if getattr(g, "cartDims", None) is None:
        return None
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    index_map = getattr(g.cells, "indexMap", None)
    box = (np.arange(g.cells.num) if index_map is None
           else np.asarray(index_map, dtype=np.int64).ravel())
    ijk = np.array(np.unravel_index(box, dims, order="F")).T
    lines = []
    for d in range(len(g.cartDims)):
        nodes = np.unique(g.nodes.coords[:, d])
        if nodes.size != dims[d] + 1:
            return None
        edges = [nodes[0]]
        for idx in range(1, dims[d]):
            here = np.flatnonzero(ijk[:, d] == idx)
            there = np.flatnonzero(ijk[:, d] == idx - 1)
            if here.size == 0 or there.size == 0:
                return None
            if partition[there[0]] != partition[here[0]]:
                edges.append(nodes[idx])
        edges.append(nodes[-1])
        lines.append(np.asarray(edges))

    cg = compute_geometry(tensor_grid(*lines))
    if cg.cells.num != n_blocks:
        return None
    cg.parent = g
    cg.partition = partition
    return cg


def generate_coarse_grid(g: Grid, partition: np.ndarray,
                         face_partition: np.ndarray | None = None) -> CoarseGrid:
    """Build the coarse grid implied by any partition of `g`.

    `face_partition` is MRST's optional `pf`: an extra indicator per fine face
    that lets one block pair carry several coarse faces. Pass it through
    `process_face_partition` first if each coarse face must be connected.
    """
    partition = np.asarray(partition, dtype=np.int64).ravel()
    if partition.size != int(g.cells.num):
        raise ValueError(
            f"partition has {partition.size} entries for {g.cells.num} cells")
    if face_partition is not None:
        pf = np.asarray(face_partition)
        if pf.shape[0] != int(g.faces.num):
            raise ValueError(
                f"face_partition has {pf.shape[0]} entries for {g.faces.num} faces")
    n_blocks = int(partition.max()) + 1

    nb, conn_pos, fconn, face_pos, faces, ind = _coarse_topology(
        g, partition, n_blocks, face_partition)
    return CoarseGrid(
        g=_tensor_coarse_grid(g, partition, n_blocks),
        partition=partition, n_blocks=n_blocks, parent=g,
        neighbors=nb, conn_pos=conn_pos, fconn=fconn,
        face_pos=face_pos, faces=faces, face_indicator=ind,
        griddim=int(getattr(g, "griddim", 3)),
    )


__all__ = ["CoarseGrid", "generate_coarse_grid"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(6, 4, 2), part=(3, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        return g, partition_cart_grid(list(dims), list(part))

    def test_coarse_cell_count_matches_the_partition():
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert cg.n_blocks == p.max() + 1 == 6, cg.n_blocks
        assert cg.g.cells.num == cg.n_blocks, cg.g.cells.num
        print(f"ok test_coarse_cell_count_matches_the_partition ({cg.n_blocks})")

    def test_volume_is_conserved():
        """Summed fine volumes must equal the coarse block volume, exactly.

        This is the invariant a coarse grid exists to preserve. A wrong
        fine-to-coarse mapping still produces a grid of the right shape, and
        only the volumes reveal it.
        """
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        fine = np.asarray(g.cells.volumes, dtype=float)
        coarse = np.asarray(cg.g.cells.volumes, dtype=float)
        assert np.isclose(fine.sum(), coarse.sum(), rtol=1e-12), (
            fine.sum(), coarse.sum())
        for b, cells in enumerate(cg.blocks()):
            assert np.isclose(fine[cells].sum(), coarse[b], rtol=1e-12), b
        print("ok test_volume_is_conserved")

    def test_blocks_partition_the_fine_cells():
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        members = np.concatenate(cg.blocks())
        assert members.size == g.cells.num
        assert np.array_equal(np.sort(members), np.arange(g.cells.num))
        print("ok test_blocks_partition_the_fine_cells")

    def test_blocks_agree_with_the_partition():
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        for b, cells in enumerate(cg.blocks()):
            assert np.all(p[cells] == b), (b, p[cells])
        print("ok test_blocks_agree_with_the_partition")

    def test_the_parent_grid_is_kept():
        """Coarse-grid plotting expands a coarse face into its fine faces.

        Discarding the parent during generation makes that impossible, so the
        link is asserted rather than assumed.
        """
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert cg.parent is g, "the fine grid was not retained"
        assert np.array_equal(np.asarray(cg.partition), p)
        print("ok test_the_parent_grid_is_kept")

    def test_a_single_block_reproduces_the_bounding_box():
        g, p = _case(part=(1, 1, 1))
        cg = generate_coarse_grid(g, p)
        assert cg.n_blocks == 1
        assert np.isclose(float(np.asarray(cg.g.cells.volumes)[0]),
                          float(np.sum(g.cells.volumes)), rtol=1e-12)
        print("ok test_a_single_block_reproduces_the_bounding_box")

    def test_full_resolution_reproduces_the_fine_grid():
        g, p = _case(dims=(3, 2, 2), part=(3, 2, 2))
        cg = generate_coarse_grid(g, p)
        assert cg.n_blocks == g.cells.num
        assert np.allclose(np.sort(np.asarray(cg.g.cells.volumes)),
                           np.sort(np.asarray(g.cells.volumes)))
        print("ok test_full_resolution_reproduces_the_fine_grid")

    def test_non_rectangular_partitions_now_build():
        """A partition that is not a tensor product still gives a coarse grid.

        This used to raise, which meant no coarse grid could be built for a
        faulted model at all. The topology is complete; only the optional
        geometric `.g` is unavailable.
        """
        g, _ = _case()
        p = np.zeros(g.cells.num, dtype=np.int64)
        p[0] = 1                      # one stray cell: not a box
        cg = generate_coarse_grid(g, p)
        assert cg.n_blocks == 2, cg.n_blocks
        assert cg.g is None, "a tensor grid was built for a non-tensor partition"
        assert cg.num_faces > 0 and cg.fconn is not None
        print("ok test_non_rectangular_partitions_now_build")

    def test_coarse_faces_separate_one_block_pair():
        """Every fine face behind a coarse face must join the same two blocks.

        This is what makes a coarse face meaningful. Grouping faces that join
        different pairs would still produce a plausible-looking grid.
        """
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        for f in range(cg.num_faces):
            sub = cg.sub_faces(f)
            assert sub.size > 0, f
            pairs = np.sort(np.where(nb[sub] >= 0, p[np.maximum(nb[sub], 0)], -1),
                            axis=1)
            assert np.all(pairs == pairs[0]), (f, pairs)
            assert np.array_equal(np.sort(cg.neighbors[f]), pairs[0]), f
        print(f"ok test_coarse_faces_separate_one_block_pair ({cg.num_faces} faces)")

    def test_intra_block_faces_are_not_coarse_faces():
        """Faces inside a block vanish into it; they are not coarse faces."""
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(nb >= 0, axis=1)
        inside = np.flatnonzero(interior & (p[nb[:, 0]] == p[nb[:, 1]]))
        assert inside.size > 0, "the case has no intra-block faces to check"
        assert not set(inside.tolist()) & set(cg.fconn.tolist())
        print(f"ok test_intra_block_faces_are_not_coarse_faces ({inside.size} dropped)")

    def test_each_fine_face_appears_at_most_once():
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert np.unique(cg.fconn).size == cg.fconn.size, "a fine face was reused"
        print("ok test_each_fine_face_appears_at_most_once")

    def test_block_to_face_mapping_is_consistent():
        """`face_pos`/`faces` must agree with `neighbors`, read the other way."""
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert cg.face_pos.size == cg.n_blocks + 1
        for b in range(cg.n_blocks):
            owned = cg.faces[cg.face_pos[b]:cg.face_pos[b + 1]]
            assert owned.size > 0, b
            for f in owned:
                assert b in cg.neighbors[f].tolist(), (b, f, cg.neighbors[f])
        # and every face is claimed by each of its real neighbours
        total = sum(int(np.count_nonzero(cg.neighbors[f] >= 0))
                    for f in range(cg.num_faces))
        assert total == int(cg.face_pos[-1]), (total, int(cg.face_pos[-1]))
        print("ok test_block_to_face_mapping_is_consistent")

    def test_a_block_boundary_is_split_by_direction():
        """MRST's default indicator carries the exterior half-face tag.

        Grouping on the block pair alone gives each block one boundary coarse
        face, which is a perfectly consistent coarse grid and the wrong one:
        MATLAB reports 29 coarse faces here, not 13. Nothing downstream raises,
        the boundary conditions simply land on the wrong connections.
        """
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert cg.num_faces == 29, cg.num_faces
        exterior = np.flatnonzero(cg.neighbors[:, 0] < 0)
        assert exterior.size == 22, exterior.size
        # Block 0 is a corner block: west, south, bottom and top.
        own = [f for f in exterior if 0 in cg.neighbors[f].tolist()]
        assert len(own) == 4, own
        assert len(set(int(cg.face_indicator[f, 0]) for f in own)) == 4
        print(f"ok test_a_block_boundary_is_split_by_direction "
              f"({exterior.size} boundary faces for {cg.n_blocks} blocks)")

    def test_a_face_partition_splits_a_block_pair():
        """MRST's optional `pf`: several coarse faces between the same blocks."""
        g, p = _case()
        plain = generate_coarse_grid(g, p)
        # Split every fine face by its own index parity.
        pf = (np.arange(g.faces.num) % 2).astype(np.int64)
        split = generate_coarse_grid(g, p, pf)
        assert split.num_faces > plain.num_faces, (plain.num_faces,
                                                   split.num_faces)
        assert split.fconn.size == plain.fconn.size, "fine faces were lost"
        for f in range(split.num_faces):
            sub = split.sub_faces(f)
            assert len(set((pf[sub]).tolist())) == 1, f
        print(f"ok test_a_face_partition_splits_a_block_pair "
              f"({plain.num_faces} became {split.num_faces})")

    def test_a_wrong_length_face_partition_is_rejected():
        g, p = _case()
        try:
            generate_coarse_grid(g, p, np.zeros(g.faces.num - 1, dtype=np.int64))
        except ValueError as exc:
            assert "entries for" in str(exc), exc
        else:
            raise AssertionError("expected a wrong-length face partition to fail")
        print("ok test_a_wrong_length_face_partition_is_rejected")

    def test_the_coarse_grid_can_be_used_as_a_grid():
        """MRST's claim that a coarse grid works in ordinary solvers.

        `compute_trans` needs centroids, normals and the packed face arrays,
        and reads them from the same fields a fine grid uses. If `to_grid`
        mislays any of them the failure is an exception, not a wrong number.
        """
        from pymrst.params.rock.make_rock import make_rock
        from pymrst.solvers.compute_trans import compute_trans

        g, p = _case()
        cg = generate_coarse_grid(g, p)
        coarse = cg.to_grid()
        assert coarse.cells.num == cg.n_blocks
        assert coarse.faces.num == cg.num_faces
        assert coarse.cells.facePos[-1] == coarse.cells.faces.size
        rock = make_rock(coarse, perm=np.ones(coarse.cells.num), poro=0.3)
        t = compute_trans(coarse, rock)
        assert t.size == coarse.cells.faces.size
        assert np.all(t > 0.0), t.min()
        print(f"ok test_the_coarse_grid_can_be_used_as_a_grid "
              f"({t.size} half-face transmissibilities)")

    def test_the_coarse_grid_view_carries_the_coarse_geometry():
        """Volumes must be the blocks', not the fine cells'."""
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        coarse = cg.to_grid()
        assert np.allclose(coarse.cells.volumes, cg.cell_volumes)
        assert np.isclose(float(coarse.cells.volumes.sum()),
                          float(np.sum(g.cells.volumes)), rtol=1e-12)
        print("ok test_the_coarse_grid_view_carries_the_coarse_geometry")

    def test_the_coarse_grid_view_has_no_nodes():
        """Recorded because it is a real limit, not an oversight.

        A block is defined by the cells it holds, not by a polygon, so there
        is no vertex list. MRST's coarse grid has none either. Anything
        needing vertices goes through `parent` and `sub_faces`.
        """
        g, p = _case()
        coarse = generate_coarse_grid(g, p).to_grid()
        assert coarse.nodes.num == 0
        assert coarse.faces.nodes.size == 0
        assert coarse.parent is g, "the fine grid must stay reachable"
        print("ok test_the_coarse_grid_view_has_no_nodes")

    def test_sub_faces_needs_connectivity():
        g, p = _case()
        cg = generate_coarse_grid(g, p)
        assert cg.sub_faces(0).size > 0
        cg.fconn = None
        try:
            cg.sub_faces(0)
        except ValueError as exc:
            assert "connectivity" in str(exc), exc
        else:
            raise AssertionError("expected sub_faces to refuse without fconn")
        print("ok test_sub_faces_needs_connectivity")

    test_coarse_cell_count_matches_the_partition()
    test_volume_is_conserved()
    test_blocks_partition_the_fine_cells()
    test_blocks_agree_with_the_partition()
    test_the_parent_grid_is_kept()
    test_a_single_block_reproduces_the_bounding_box()
    test_full_resolution_reproduces_the_fine_grid()
    test_non_rectangular_partitions_now_build()
    test_coarse_faces_separate_one_block_pair()
    test_intra_block_faces_are_not_coarse_faces()
    test_each_fine_face_appears_at_most_once()
    test_block_to_face_mapping_is_consistent()
    test_a_block_boundary_is_split_by_direction()
    test_a_face_partition_splits_a_block_pair()
    test_a_wrong_length_face_partition_is_rejected()
    test_the_coarse_grid_can_be_used_as_a_grid()
    test_the_coarse_grid_view_carries_the_coarse_geometry()
    test_the_coarse_grid_view_has_no_nodes()
    test_sub_faces_needs_connectivity()
    print("\nAll tests passed!")
