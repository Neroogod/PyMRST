%% Reference data for markCutGrids

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([4, 2, 1]));
face_status = zeros(G.faces.num, 1);
N = G.faces.neighbors;
internal = all(N > 0, 2);
left_i = mod(max(N(:, 1), 1) - 1, 4);
right_i = mod(max(N(:, 2), 1) - 1, 4);
face_status(internal & left_i == 1 & right_i == 2) = 3;

labels = markCutGrids(G, face_status);
legacy_left = markCutGrids( ...
    G, face_status, 'legacy', true, 'start', G.cells.centroids(1, :));
legacy_right = markCutGrids( ...
    G, face_status, 'legacy', true, 'start', G.cells.centroids(4, :));

save(fullfile(thisDir, 'ref_grid_mark_cut_grids.mat'), '-v7', ...
     'face_status', 'labels', 'legacy_left', 'legacy_right');
fprintf('Wrote markCutGrids reference to %s\n', thisDir);
