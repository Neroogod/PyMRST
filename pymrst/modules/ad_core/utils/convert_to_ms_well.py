"""PyMRST's earlier multisegment conversion (0-based topology, its own option
names). MRST's `convert2MSWell` is ported in `convert2_ms_well.py`."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp


def _cell_to_node_matrix(mapping, number_of_perforations):
    if mapping is None:
        return sp.eye(number_of_perforations, format="csr")
    if sp.issparse(mapping):
        matrix = mapping.tocsr()
    else:
        array = np.asarray(mapping)
        if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
            nodes = array.astype(np.int64).ravel()
            if nodes.size != number_of_perforations:
                raise ValueError("cell2node vector must have one entry per perforation")
            matrix = sp.csr_matrix(
                (
                    np.ones(number_of_perforations),
                    (nodes, np.arange(number_of_perforations)),
                ),
                shape=(int(nodes.max()) + 1, number_of_perforations),
            )
        else:
            matrix = sp.csr_matrix(array)
    if matrix.shape[1] != number_of_perforations:
        raise ValueError("cell2node must have one column per perforation")
    return matrix


def convert_to_ms_well(
    well,
    *,
    cell2node=None,
    conn_dz=None,
    node_depth=None,
    topology=None,
    segment_length=None,
    segment_roughness=None,
    segment_flow_model=None,
    segment_type=None,
    segment_diameter=None,
    grid=None,
    volume=None,
):
    """Return a standard well augmented with nodes and connecting segments.

    Python node/topology indices are zero-based. A vector ``cell2node`` maps
    each perforation to its node; a matrix has shape ``(nodes, perforations)``.
    """
    output = deepcopy(well)
    cells = np.asarray(output.cells, dtype=np.int64).ravel()
    number_of_perforations = cells.size
    mapping = _cell_to_node_matrix(cell2node, number_of_perforations)
    output.cell2node = mapping
    counts = np.maximum(np.asarray(mapping.sum(axis=1)).ravel(), 1.0)
    number_of_nodes = mapping.shape[0]

    if node_depth is None:
        depths = (
            float(output.refDepth)
            + np.asarray(mapping @ np.asarray(output.dZ, dtype=float)).ravel()
            / counts
        )
    else:
        depths = np.asarray(node_depth, dtype=float).ravel()
    volumes = (
        np.ones(number_of_nodes)
        if volume is None
        else np.asarray(volume, dtype=float).ravel()
    )
    output.nodes = SimpleNamespace(
        depth=depths,
        vol=volumes,
        dist=np.full(number_of_nodes, np.nan),
    )
    output.connDZ = (
        np.zeros_like(np.asarray(output.dZ, dtype=float))
        if conn_dz is None
        else np.asarray(conn_dz, dtype=float)
    )

    topo = (
        np.column_stack([
            np.arange(max(number_of_nodes - 1, 0), dtype=np.int64),
            np.arange(1, number_of_nodes, dtype=np.int64),
        ])
        if topology is None
        else np.asarray(topology, dtype=np.int64).reshape(-1, 2)
    )
    number_of_segments = topo.shape[0]
    if segment_length is None:
        if grid is None:
            raise ValueError("grid is required when segment_length is not supplied")
        centroids = np.asarray(grid.cells.centroids)[cells]
        node_centroids = np.asarray(mapping @ centroids) / counts[:, None]
        differences = node_centroids[topo[:, 1]] - node_centroids[topo[:, 0]]
        lengths = np.linalg.norm(differences, axis=1)
    else:
        lengths = np.asarray(segment_length, dtype=float).ravel()

    if segment_diameter is None:
        radii = np.asarray(output.r, dtype=float).ravel()
        if radii.size == 1:
            radii = np.full(number_of_perforations, radii[0])
        node_radii = np.asarray(mapping @ radii).ravel() / counts
        diameters = node_radii[topo[:, 1]] + node_radii[topo[:, 0]]
    else:
        diameters = np.asarray(segment_diameter, dtype=float).ravel()
    roughness = (
        np.zeros(number_of_segments)
        if segment_roughness is None
        else np.asarray(segment_roughness, dtype=float).ravel()
    )
    flow_model = (
        np.zeros(number_of_segments)
        if segment_flow_model is None
        else np.asarray(segment_flow_model).ravel()
    )
    delta_distance = np.zeros(number_of_segments)
    if segment_type is not None:
        types = np.asarray(segment_type).ravel()
        for segment in np.flatnonzero(types == 3):
            connected_nodes = topo[segment]
            local = np.flatnonzero(np.any(np.isin(topo, connected_nodes), axis=1))
            tubing = local[types[local] == 1]
            if tubing.size == 0:
                raise ValueError("valve segment must neighbor at least one tubing segment")
            delta_distance[segment] = np.mean(lengths[tubing])
    output.segments = SimpleNamespace(
        length=lengths,
        roughness=roughness,
        diam=diameters,
        flowModel=flow_model,
        topo=topo,
        deltaDistance=delta_distance,
    )
    output.isMS = True
    return output


__all__ = ["convert_to_ms_well"]
