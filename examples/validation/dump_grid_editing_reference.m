%% Reference data for extractSubgrid, removeCells, splitDisconnectedGrid,
%% makeInternalBoundary and removeInternalBoundary.

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([5, 1, 1], [5, 1, 1]));

E = extractSubgrid(G, [1, 2, 4, 5]);
extract_counts = [E.cells.num, E.faces.num, E.nodes.num];
extract_index_map = E.cells.indexMap;
extract_volumes = E.cells.volumes;
extract_centroids = E.cells.centroids;
extract_face_areas = E.faces.areas;
extract_face_centroids = E.faces.centroids;

[R, remove_cell_map, remove_face_map, remove_node_map] = removeCells(G, 3);
remove_counts = [R.cells.num, R.faces.num, R.nodes.num];
remove_index_map = R.cells.indexMap;
remove_volumes = R.cells.volumes;
remove_centroids = R.cells.centroids;
remove_face_areas = R.faces.areas;
remove_face_centroids = R.faces.centroids;

S = splitDisconnectedGrid(R);
split_counts = arrayfun(@(g) g.cells.num, S).';
split_pos = cumsum([1; split_counts]);
split_index_maps = vertcat(S.cells);
split_index_maps = vertcat(split_index_maps.indexMap);

cut = find(all(G.faces.neighbors > 0, 2) & ...
           any(G.faces.neighbors == 2, 2) & ...
           any(G.faces.neighbors == 3, 2), 1);
[B, N] = makeInternalBoundary(G, cut);
[W, merged] = removeInternalBoundary(B, N);
roundtrip_counts = [W.cells.num, W.faces.num, W.nodes.num];
roundtrip_neighbors = sort(W.faces.neighbors, 2);
roundtrip_neighbors = sortrows(roundtrip_neighbors);
roundtrip_merged_neighbors = sort(W.faces.neighbors(merged, :), 2);

save(fullfile(thisDir, 'ref_grid_editing.mat'), '-v7', ...
     'extract_counts', 'extract_index_map', 'extract_volumes', ...
     'extract_centroids', 'extract_face_areas', 'extract_face_centroids', ...
     'remove_counts', 'remove_index_map', 'remove_volumes', ...
     'remove_centroids', 'remove_face_areas', 'remove_face_centroids', ...
     'remove_cell_map', 'remove_face_map', 'remove_node_map', ...
     'split_counts', 'split_pos', 'split_index_maps', ...
     'roundtrip_counts', 'roundtrip_neighbors', ...
     'roundtrip_merged_neighbors');
fprintf('Wrote grid-editing reference to %s\n', thisDir);
