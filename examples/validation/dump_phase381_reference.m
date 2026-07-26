%% Dump Phase 381 reference: removePinch (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p381_pinch.mat.
%
% Pins the PyMRST `remove_pinch` port against MRST's `removePinch`. Two unit
% squares are concatenated *with* their shared-edge nodes and faces duplicated
% (the situation a pinched layer produces), then removePinch is run. The
% resulting cell/face/node counts and sorted neighbour-pair multiset are compared
% - both codes must merge the duplicates and reconnect the two cells.

thisDir = fileparts(mfilename('fullpath'));

g1 = tessellationGrid([0 0; 1 0; 1 1; 0 1], [1 2 3 4]);
g2 = tessellationGrid([1 0; 2 0; 2 1; 1 1], [1 2 3 4]);

% concatenate without deduplication (as glue2DGrid's concatenate_grids does)
G.nodes.coords = [g1.nodes.coords; g2.nodes.coords];
G.nodes.num    = g1.nodes.num + g2.nodes.num;
G.faces.num    = g1.faces.num + g2.faces.num;
G.faces.nodePos = (1:G.faces.num + 1)' * 2 - 1;
neigh2 = g2.faces.neighbors; neigh2(neigh2 ~= 0) = neigh2(neigh2 ~= 0) + g1.cells.num;
G.faces.neighbors = [g1.faces.neighbors; neigh2];
G.faces.nodes  = [g1.faces.nodes; g2.faces.nodes + g1.nodes.num];
G.cells.num    = g1.cells.num + g2.cells.num;
G.cells.facePos = [g1.cells.facePos; g2.cells.facePos(2:end) + g1.cells.facePos(end) - 1];
G.cells.faces  = [g1.cells.faces(:,1); g2.cells.faces(:,1) + g1.faces.num];
G.griddim = 2;
G.type = {'concat'};

G = removePinch(G);

cells_num = G.cells.num;
faces_num = G.faces.num;
nodes_num = G.nodes.num;
nb = double(G.faces.neighbors); nb(nb == 0) = -1;
pairs = sortrows(sort(nb, 2));

save(fullfile(thisDir, 'ref_p381_pinch.mat'), '-v7', ...
     'cells_num', 'faces_num', 'nodes_num', 'pairs');

fprintf('Wrote ref_p381_pinch.mat to %s\n', thisDir);
fprintf('  concat(8 nodes,8 faces) -> removePinch: %d cells, %d faces, %d nodes\n', ...
        cells_num, faces_num, nodes_num);
