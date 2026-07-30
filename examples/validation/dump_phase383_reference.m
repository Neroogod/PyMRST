%% Dump Phase 383 reference: sliceGrid cell-splitting (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p383_slice.mat.
%
% Pins the PyMRST `slice_grid` clean-cut port against MRST's `sliceGrid` for a
% vertical plane x=1.5 cutting a 3x2x2 grid. The plane splits the four cells of
% the middle column. Cell renumbering differs, so the check uses the cell count,
% the total volume, and the sorted per-cell volumes - which pin the geometry of
% the split.

mrstModule add libgeometry

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 2, 2], [3, 2, 2]));

% segment overspans y so the finite-segment clamping is a no-op
coord = [1.5 -1 0; 1.5 3 0];
[G2, gix] = sliceGrid(G, coord, 'cutDir', [0 0 1]);
G2 = computeGeometry(G2);

cells_num   = G2.cells.num;
faces_num   = G2.faces.num;
total_vol   = sum(G2.cells.volumes);
vol_sorted  = sort(G2.cells.volumes(:));
old_cell_count = nnz(gix.old.cells);
old_face_count = nnz(gix.old.faces);
new_cell_status_counts = accumarray(gix.new.cells, 1, [3, 1]);
new_face_status_counts = accumarray(gix.new.faces, 1, [3, 1]);
parent_cells = gix.parent.cells;
face_parent_status = sortrows([gix.parent.faces, gix.new.faces]);

save(fullfile(thisDir, 'ref_p383_slice.mat'), '-v7', ...
     'cells_num', 'faces_num', 'total_vol', 'vol_sorted', ...
     'old_cell_count', 'old_face_count', 'new_cell_status_counts', ...
     'new_face_status_counts', 'parent_cells', 'face_parent_status');

fprintf('Wrote ref_p383_slice.mat to %s\n', thisDir);
fprintf('  3x2x2 sliced at x=1.5: %d cells, total vol %.4f\n', ...
        cells_num, total_vol);
