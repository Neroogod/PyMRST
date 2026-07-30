%% Dump Phase 380 reference: cellNodes (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p380_cellnodes.mat.
%
% Pins the PyMRST `cell_nodes` port against MRST's `cellNodes` on a 2x2x1
% Cartesian grid. The full [cell, local-vertex, node] table is dumped; PyMRST
% reproduces it (converted from MRST's 1-based indices) number for number,
% confirming the direction-tag binary-key corner identification matches.

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([2, 2, 1], [2, 2, 1]));
s = cellNodes(G);              % [cellNumber, localVertex(1..8), nodeNumber]

G1 = tensorGrid([0, 1, 3, 6], 'cellnodes', true);
G2 = tensorGrid([0, 1, 2], [0, 2, 5], 'cellnodes', true);
G3 = tensorGrid([0, 1, 2], [0, 2, 5], [0, 4], 'cellnodes', true);
tensor_cellnodes_1d = G1.cellNodes;
tensor_cellnodes_2d = G2.cellNodes;
tensor_cellnodes_3d = G3.cellNodes;

save(fullfile(thisDir, 'ref_p380_cellnodes.mat'), '-v7', 's', ...
     'tensor_cellnodes_1d', 'tensor_cellnodes_2d', 'tensor_cellnodes_3d');

fprintf('Wrote ref_p380_cellnodes.mat to %s\n', thisDir);
fprintf('  %d corner rows for %d cells\n', size(s, 1), G.cells.num);
