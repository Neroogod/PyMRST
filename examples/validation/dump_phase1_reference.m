%% Dump Phase 1 reference grids from MATLAB MRST for PyMRST validation.
%
% Run this script in MATLAB *after* starting MRST (run startup.m in
% mrst-2024a-Geomech). It writes one .mat file per reference case into
% this folder. Then run validate_phase1.py to compare PyMRST against them.

thisDir = fileparts(mfilename('fullpath'));

%% Case 1: uniform 3D Cartesian grid
G = computeGeometry(cartGrid([3, 4, 5], [300, 200, 100]));
save(fullfile(thisDir, 'ref_cart_3x4x5.mat'), '-v7', 'G');

%% Case 2: graded tensor grid with depthz
x = [0, 1, 2.5, 5, 10];
y = [0, 2, 3, 7];
z = [0, 1, 4];
depthz = reshape(0.1 * (0 : numel(x) * numel(y) - 1), numel(x), numel(y));
G = computeGeometry(tensorGrid(x, y, z, 'depthz', depthz));
save(fullfile(thisDir, 'ref_tensor_graded.mat'), '-v7', 'G');

%% Case 3: 2D Cartesian grid
G = computeGeometry(cartGrid([5, 7], [10, 21]));
save(fullfile(thisDir, 'ref_cart_2d_5x7.mat'), '-v7', 'G');

%% Case 4: 1D tensor grid
G = computeGeometry(tensorGrid([0, 1, 2.5, 5]));
save(fullfile(thisDir, 'ref_tensor_1d.mat'), '-v7', 'G');

%% Case 5: independently truncated non-monotone tensor axes
G = computeGeometry(tensorGrid( ...
    [0, 2, 1, 3], [0, 1, 1, 4], [0, 5, 3, 8]));
save(fullfile(thisDir, 'ref_tensor_nonmonotone.mat'), '-v7', 'G');

%% Cases 6/7: reconstruct missing face neighbors in 2D and 3D
G = cartGrid([3, 2], [6, 4]);
G.faces = rmfield(G.faces, 'neighbors');
G = computeGeometry(G, 'findNeighbors', true);
save(fullfile(thisDir, 'ref_find_neighbors_2d.mat'), '-v7', 'G');

G = cartGrid([3, 2, 2], [6, 4, 2]);
G.faces = rmfield(G.faces, 'neighbors');
G = computeGeometry(G, 'findNeighbors', true);
save(fullfile(thisDir, 'ref_find_neighbors_3d.mat'), '-v7', 'G');

fprintf('Wrote 7 reference files to %s\n', thisDir);
