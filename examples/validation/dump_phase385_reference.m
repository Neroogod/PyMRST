%% Dump Phase 385 reference: sliceGrid partial cut, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p385_slicepartial.mat.
%
% Pins the PyMRST `slice_grid` partial-cut path against MRST's `sliceGrid`. A
% segment that stops at x=2.5 cuts cells 0 and 1 of a 3x1x1 grid but not cell 2
% (the segment ends inside it): MRST removes partial cuts. Cell count and sorted
% per-cell volumes are compared.

mrstModule add libgeometry

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));

coord = [0 0.5 0; 2.5 0.5 0];          % plane y=0.5, window x in [0, 2.5]
G2 = sliceGrid(G, coord, 'cutDir', [0 0 1]);
G2 = computeGeometry(G2);

cells_num  = G2.cells.num;
total_vol  = sum(G2.cells.volumes);
vol_sorted = sort(G2.cells.volumes(:));

save(fullfile(thisDir, 'ref_p385_slicepartial.mat'), '-v7', ...
     'cells_num', 'total_vol', 'vol_sorted');

fprintf('Wrote ref_p385_slicepartial.mat to %s\n', thisDir);
fprintf('  3x1x1 partial cut (window x<=2.5): %d cells, total vol %.4f\n', ...
        cells_num, total_vol);
