%% Dump Phase 107 reference: MRST HFM 3D planar EDFM preprocessing.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrst_root = fullfile(root, '..', 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add hfm ad-core incomp coarsegrid msrsb

celldim = [4, 4, 3];
physdim = [4, 4, 3];
G = cartGrid(celldim, physdim);
G = computeGeometry(G);
GlobTri = globalTriangulation(G);

fracplanes = struct;
fracplanes(1).points = [2.0 0.5 0.5;
                        2.0 0.5 2.5;
                        2.0 3.5 2.5;
                        2.0 3.5 0.5];
fracplanes(1).aperture = 0.02;
fracplanes(1).perm = 10000*darcy();
fracplanes(1).poro = 0.5;
checkIfCoplanar(fracplanes);

[G, fracplanes] = preProcessingFractures(G, fracplanes, ...
    'GlobTri', GlobTri, 'fractureCellSize', 0.35, 'Verbose', false);

F = G.FracGrid.Frac1;
matrix_cells = G.nnc.cells(:, 1);
matrix_area = G.nnc.area(:);
matrix_ci = G.nnc.CI(:);
frac_volumes = F.cells.volumes(:);
frac_centroids = F.cells.centroids;
points = fracplanes(1).points;

save(fullfile(fileparts(mfilename('fullpath')), 'ref_p107_hfm3d.mat'), ...
    'celldim', 'physdim', 'points', 'matrix_cells', 'matrix_area', ...
    'matrix_ci', 'frac_volumes', 'frac_centroids');

fprintf('Wrote Phase 107 HFM 3D reference: %d matrix intersections, %d fracture cells\n', ...
    numel(matrix_cells), numel(frac_volumes));
