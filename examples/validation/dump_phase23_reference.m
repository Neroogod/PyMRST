%% Dump Phase 23 reference grids (unstructured constructors) from MATLAB MRST.
%
% Run this script in MATLAB *after* starting MRST (run startup.m in
% mrst-2024a-Geomech). It writes one .mat file per reference case into
% this folder. Then run validate_phase23.py to compare PyMRST against them.
%
% The point sets and triangulations are saved alongside each grid, so the
% Python side rebuilds from the *same* connectivity and what is compared
% is the grid construction, not the triangulator.

thisDir = fileparts(mfilename('fullpath'));

%% Case 1: triangleGrid on the honeycomb lattice from the MRST docstring
N  = 4;
N1 = 2 * N;
N2 = 3 * ceil(N / 2) - 2;
[X, Y] = meshgrid(0 : 1 : N1, 0 : 1 : N2);
X = sqrt(3) / 2 * X;
Y(:, 2 : 2 : end) = Y(:, 2 : 2 : end) + 0.5;
p = [X(:), Y(:)];
t = delaunayn(p);
G = computeGeometry(triangleGrid(p, t));
save(fullfile(thisDir, 'ref_p23_tri.mat'), '-v7', 'G', 'p', 't');

%% Case 2: pebi (Voronoi dual) of the honeycomb triangulation
G = computeGeometry(pebi(triangleGrid(p, t)));
save(fullfile(thisDir, 'ref_p23_pebi.mat'), '-v7', 'G', 'p', 't');

%% Case 3: pebi of a squashed lattice whose boundary-triangle circumcenters
%% fall outside the domain - exercises the auxiliary-triangle path
[Xb, Yb] = meshgrid(0 : 6, 0 : 2);
Yb = Yb * 0.35;
Yb(:, 2 : 2 : end) = Yb(:, 2 : 2 : end) + 0.10;
pb = [Xb(:), Yb(:)];
tb = delaunayn(pb);
G = computeGeometry(pebi(triangleGrid(pb, tb)));
save(fullfile(thisDir, 'ref_p23_pebi_aux.mat'), '-v7', 'G', 'pb', 'tb');

%% Case 4: tetrahedralGrid on a perturbed 3x3x3 lattice
[xx, yy, zz] = ndgrid(0:2, 0:2, 0:2);
p3 = [xx(:), yy(:), zz(:)];
interior = all(p3 > 0, 2) & all(p3 < 2, 2);
p3 = p3 + 0.15 * [sin(1 + 3 * (1:27)'), cos(2 + 5 * (1:27)'), ...
                  sin(3 + 7 * (1:27)')] .* interior;
t3 = delaunayn(p3);
G = computeGeometry(tetrahedralGrid(p3, t3));
save(fullfile(thisDir, 'ref_p23_tet.mat'), '-v7', 'G', 'p3', 't3');

%% Case 5: makeLayeredGrid extrusion of the honeycomb triangle grid
thickness = [1, 2.5, 0.5];
G = computeGeometry(makeLayeredGrid(triangleGrid(p, t), thickness));
save(fullfile(thisDir, 'ref_p23_layered.mat'), '-v7', 'G', 'p', 't', ...
     'thickness');

fprintf('Wrote 5 reference files to %s\n', thisDir);
