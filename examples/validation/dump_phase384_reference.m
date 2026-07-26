%% Dump Phase 384 reference: sliceGrid snapped (node-aligned) cut, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p384_slicesnap.mat.
%
% Pins the PyMRST `slice_grid` node-snapping path against MRST's `sliceGrid` for
% the diagonal plane x=y through a 2x2x1 grid: the plane passes exactly through
% the corner nodes, cutting the two on-diagonal cells and snapping to shared
% corners. Cell count and sorted per-cell volumes are compared.

mrstModule add libgeometry

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([2, 2, 1], [2, 2, 1]));

coord = [-1 -1 0; 3 3 0];              % x = y plane, swept in z
G2 = sliceGrid(G, coord, 'cutDir', [0 0 1]);
G2 = computeGeometry(G2);

cells_num  = G2.cells.num;
total_vol  = sum(G2.cells.volumes);
vol_sorted = sort(G2.cells.volumes(:));

save(fullfile(thisDir, 'ref_p384_slicesnap.mat'), '-v7', ...
     'cells_num', 'total_vol', 'vol_sorted');

fprintf('Wrote ref_p384_slicesnap.mat to %s\n', thisDir);
fprintf('  2x2x1 sliced at x=y (snapped): %d cells, total vol %.4f\n', ...
        cells_num, total_vol);
