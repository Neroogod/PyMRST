%% Dump Phase 364 reference: flux<->velocity conversions (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p364_flux.mat.
%
% Phase 25 core leftovers: faceFlux2cellVelocity, faceFlux2cellFlux and
% cellFlux2faceFlux. This pins the PyMRST ports against MRST on a small 3D
% Cartesian grid with a deterministic face-flux field, so the comparison is
% exact and independent of any random draw.
%
% The grid geometry (cell and face centroids, cell volumes) is dumped too, so
% PyMRST can confirm the two codes order cells and faces identically before
% comparing the flux fields cell for cell.

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([4, 3, 2], [4, 3, 2]));
nf = G.faces.num;

% a deterministic, sign-varying face flux (no randomness to reproduce)
faceFlux = sin((1:nf)') + 0.3 * cos(0.5 * (1:nf)');

v  = faceFlux2cellVelocity(G, faceFlux);   % (ncells x dim)
cf = faceFlux2cellFlux(G, faceFlux);       % (n half-faces x 1)
ff = cellFlux2faceFlux(G, cf);             % (nfaces x 1) - averages back

cell_centroids = G.cells.centroids;
face_centroids = G.faces.centroids;
cell_volumes   = G.cells.volumes;

save(fullfile(thisDir, 'ref_p364_flux.mat'), '-v7', ...
     'faceFlux', 'v', 'cf', 'ff', ...
     'cell_centroids', 'face_centroids', 'cell_volumes');

fprintf('Wrote ref_p364_flux.mat to %s\n', thisDir);
fprintf('  ncells = %d, nfaces = %d, n half-faces = %d\n', ...
        G.cells.num, nf, numel(cf));
fprintf('  v(1,:) = [%.4f %.4f %.4f]\n', v(1, 1), v(1, 2), v(1, 3));
