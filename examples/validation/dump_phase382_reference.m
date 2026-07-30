%% Dump Phase 382 reference: computeGridSlicePolygons (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p382_slicepoly.mat.
%
% Pins the PyMRST `compute_grid_slice_polygons` core against MRST's
% `computeGridSlicePolygons` for a vertical plane sweeping a diagonal segment
% across a 3x3x2 grid. The cut cells, per-cell polygon vertex counts and the
% sorted set of 3D intersection coordinates are compared (patch output,
% removeRepeated on).

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3, 2], [3, 3, 2]));

% plane x = 1.5, swept in z; segment overspans y so the finite-segment
% clamping (not ported in the core) is a no-op
coord  = [1.5 -0.5 0; 1.5 3.5 0];
poly = computeGridSlicePolygons(G, coord, 'cutDir', [0;0;1], ...
                                'removeRepeated', true, ...
                                'outputForGridProcessing', false);

cell_ix = poly.cellIx(:);
% per-polygon vertex count from the (padded) nodes matrix
nvert = arrayfun(@(r) numel(unique(poly.nodes(r,:))), (1:size(poly.nodes,1))');
coords3d = sortrows(round(poly.coords3D, 6));

Gm = computeGeometry(cartGrid([3, 3, 1], [3, 3, 1]));
multi_coord = [0.5, 0, 0; 0.5, 1.5, 0; 2.5, 1.5, 0];
multi = computeGridSlicePolygons(Gm, multi_coord, 'cutDir', [0;0;1], ...
                                 'removeRepeated', true);
multi_cell_ix = multi.cellIx(:);
multi_nvert = arrayfun(@(r) numel(unique(multi.nodes(r,:))), ...
                       (1:size(multi.nodes,1))');
multi_coords3d = sortrows(round(multi.coords3D, 6));

Gr = computeGeometry(cartGrid([1, 1, 1], [1, 1, 1]));
radius_coord = [0.5, 0, 0.5; 0.5, 1, 0.5];
bounded = computeGridSlicePolygons(Gr, radius_coord, 'cutDir', [0;0;1], ...
                                   'radius', 0.2, 'removeRepeated', true);
radius_cell_ix = bounded.cellIx(:);
radius_nvert = arrayfun(@(r) numel(unique(bounded.nodes(r,:))), ...
                        (1:size(bounded.nodes,1))');
radius_coords3d = sortrows(round(bounded.coords3D, 6));

save(fullfile(thisDir, 'ref_p382_slicepoly.mat'), '-v7', ...
     'cell_ix', 'nvert', 'coords3d', 'multi_cell_ix', 'multi_nvert', ...
     'multi_coords3d', 'radius_cell_ix', 'radius_nvert', 'radius_coords3d');

fprintf('Wrote ref_p382_slicepoly.mat to %s\n', thisDir);
fprintf('  %d cells cut, %d total vertices\n', numel(cell_ix), size(coords3d,1));
