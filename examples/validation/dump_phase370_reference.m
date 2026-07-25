%% Dump Phase 370 reference: hexahedralGrid (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p370_hexgrid.mat.
%
% Pins the PyMRST `hexahedral_grid` port against MRST's `hexahedralGrid` on a
% deterministic two-cube mesh (two unit cubes stacked in z, sharing their mid
% plane). Face renumbering may differ, so the check uses the face/cell/node
% counts and the sorted multiset of neighbour pairs (outside encoded as -1) -
% which fully pin the assembled topology. H and P are dumped so PyMRST builds
% from the identical connectivity.

thisDir = fileparts(mfilename('fullpath'));

layer = [0 0; 1 0; 1 1; 0 1];
P = [ [layer, zeros(4,1)]; [layer, ones(4,1)]; [layer, 2*ones(4,1)] ];
a = 1:4; b = 5:8; c = 9:12;
H = [a, b; b, c];                 % 1-based node indices, two hexes

G = hexahedralGrid(P, H);         % MRST signature is (points, hexes)

faces_num = G.faces.num;
cells_num = G.cells.num;
nodes_num = G.nodes.num;

nb = double(G.faces.neighbors);
nb(nb == 0) = -1;                 % outside -> -1
nb = sort(nb, 2);
pairs = sortrows(nb);

H0 = H - 1;                       % 0-based for PyMRST

save(fullfile(thisDir, 'ref_p370_hexgrid.mat'), '-v7', ...
     'faces_num', 'cells_num', 'nodes_num', 'pairs', 'H0', 'P');

fprintf('Wrote ref_p370_hexgrid.mat to %s\n', thisDir);
fprintf('  cells %d, faces %d, nodes %d\n', cells_num, faces_num, nodes_num);
