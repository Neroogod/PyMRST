%% Dump Phase 61 reference: 2D VEM geometry and stiffness.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p61_vem2d.mat into this folder; then run validate_phase61.py.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

nx = 3; ny = 2;
L = [3, 2];
E = 1e9;
nu = 0.25;

G = computeGeometry(cartGrid([nx, ny], L));
G = createAugmentedGrid(G);

[qf, qf_vol] = calculateQF(G);
div = VEM2D_div(G);
C = Enu2C(E * ones(G.cells.num, 1), nu * ones(G.cells.num, 1), G);
S = VEM_assemble(G, C);

coords = G.nodes.coords;
cell_nodes = G.cells.nodes;
cell_node_pos = G.cells.nodePos;
volumes = G.cells.volumes;
centroids = G.cells.centroids;
faces_nodes = G.faces.nodes;
faces_node_pos = G.faces.nodePos;
faces_neighbors = G.faces.neighbors;
faces_centroids = G.faces.centroids;
faces_normals = G.faces.normals;
faces_areas = G.faces.areas;
cells_faces = G.cells.faces(:, 1);
cells_face_pos = G.cells.facePos;

save(fullfile(thisDir, 'ref_p61_vem2d.mat'), '-v7', ...
     'nx', 'ny', 'L', 'E', 'nu', ...
     'coords', 'cell_nodes', 'cell_node_pos', 'volumes', 'centroids', ...
     'faces_nodes', 'faces_node_pos', 'faces_neighbors', ...
     'faces_centroids', 'faces_normals', 'faces_areas', ...
     'cells_faces', 'cells_face_pos', ...
     'qf', 'qf_vol', 'div', 'C', 'S');

fprintf('Wrote ref_p61_vem2d.mat to %s\n', thisDir);
