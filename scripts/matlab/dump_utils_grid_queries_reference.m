% MRST references for topology and packed grid-query helpers.
repo = '/Users/Shared/PyMRST_workspace';
mrst = fullfile(repo, 'mrst-2024a-Geomech');
addpath(fullfile(mrst, 'utils'));
addpath(fullfile(mrst, 'utils', 'gridtools'));
addpath(fullfile(mrst, 'gridprocessing'));

G = cartGrid([3, 2], [3, 2]);
subset = [1; 5];

[cell_faces, cell_face_pos] = gridCellFaces(G, subset);
cell_no_all = gridCellNo(G);
cell_no_subset = gridCellNo(G, subset);
[cell_nodes, cell_node_pos] = gridCellNodes(G, subset);
[cell_nodes_raw, cell_node_pos_raw] = gridCellNodes(G, subset, ...
                                                   'unique', false);
face_subset = [1; G.faces.num];
[face_nodes, face_node_pos] = gridFaceNodes(G, face_subset);

logical_cells = [1; 3; 5];
[logical_i, logical_j] = gridLogicalIndices(G, logical_cells);

packed_values = [10; 11; 12; 20; 21; 30; 31; 32; 33];
packed_pos = [1; 4; 6; 10];
packed_items = [3; 1];
packed_out = indirectionSub(packed_items, packed_pos, packed_values);

geometric_n = getNeighbourship(G);
boundary_n = getNeighbourship(G, 'Geometrical', true);
connectivity = full(getConnectivityMatrix(geometric_n, true, G.cells.num));
node_n = neighboursByNodes(G);

% MRST's dmperm-to-subset remapping assumes this list is sorted.
component_cells = [1; 3; 4; 6];
components = connectedCells(G, component_cells);
component_a = sort(components{1});
component_b = sort(components{2});
if component_a(1) > component_b(1)
    tmp = component_a; component_a = component_b; component_b = tmp;
end

G.nnc.cells = [1, 6];
[nnc_cell_no, nnc_cell_faces, nnc_flag] = getCellNoFaces(G);
[topological_n, topological_flag] = getNeighbourship(G, 'Topological');

Gg = computeGeometry(cartGrid([2, 2], [2, 2]));
translated = translateGrid(Gg, [5, -2]);
translated_nodes = translated.nodes.coords;
translated_cells = translated.cells.centroids;
translated_faces = translated.faces.centroids;

destination = fullfile(repo, 'PyMRST', 'scripts', 'matlab', ...
                       'ref_utils_grid_queries.mat');
save(destination, 'subset', 'cell_faces', 'cell_face_pos', 'cell_no_all', ...
     'cell_no_subset', 'cell_nodes', 'cell_node_pos', 'cell_nodes_raw', ...
     'cell_node_pos_raw', 'face_subset', 'face_nodes', 'face_node_pos', ...
     'logical_cells', 'logical_i', 'logical_j', 'packed_values', ...
     'packed_pos', 'packed_items', 'packed_out', 'geometric_n', ...
     'boundary_n', 'connectivity', 'node_n', 'component_cells', ...
     'component_a', 'component_b', 'nnc_cell_no', 'nnc_cell_faces', ...
     'nnc_flag', 'topological_n', 'topological_flag', ...
     'translated_nodes', 'translated_cells', 'translated_faces', '-v7');
fprintf('wrote %s\n', destination);
