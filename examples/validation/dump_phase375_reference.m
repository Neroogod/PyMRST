%% Dump Phase 375 reference: get2DGridFromFaces (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p375_get2d.mat.
%
% Pins the PyMRST `get_2d_grid_from_faces` port against MRST's
% `get2DGridFromFaces`. The top-boundary faces of a 3x3x2 grid (a consistently
% oriented 3x3 surface) are turned into a 2D grid, and the resulting cell/face/
% node counts and sorted neighbour-pair multiset are compared. The selected face
% list is dumped so PyMRST builds the 2D grid from the identical faces.

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3, 2], [3, 3, 2]));

isbnd = any(G.faces.neighbors == 0, 2);
ztop  = abs(G.faces.centroids(:,3) - max(G.faces.centroids(:,3))) < 1e-9;
f     = find(isbnd & ztop);

g2 = get2DGridFromFaces(G, f);

cells_num = g2.cells.num;
faces_num = g2.faces.num;
nodes_num = g2.nodes.num;                 % NOTE: MRST sets this to the edge
                                          % count (a bug) - see coords_rows
coords_rows = size(g2.nodes.coords, 1);   % the true node count

nb = double(g2.faces.neighbors);
nb(nb == 0) = -1;
nb = sort(nb, 2);
pairs = sortrows(nb);

f0 = f - 1;         % 0-based face selection for PyMRST

save(fullfile(thisDir, 'ref_p375_get2d.mat'), '-v7', ...
     'cells_num', 'faces_num', 'nodes_num', 'coords_rows', 'pairs', 'f0');

fprintf('Wrote ref_p375_get2d.mat to %s\n', thisDir);
fprintf('  %d top faces -> 2D grid: %d cells, %d faces, nodes.num=%d, coords rows=%d\n', ...
        numel(f), cells_num, faces_num, nodes_num, coords_rows);
