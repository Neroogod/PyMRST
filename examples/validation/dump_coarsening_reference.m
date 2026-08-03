% Reference for the coarse-grid coarsening layer (modules/coarsegrid/utils).
%
% coarsenGeometry, fineToCoarseSign, coarsenFlux, invertPartition and
% refineNearWell. The geometry is the one worth the round trip: summed volumes
% and area-weighted centroids produce plausible numbers when the index arrays
% are subtly wrong.
%
% Run from the MRST root:
%   matlab -batch "startup; run('.../dump_coarsening_reference.m')"

mrstModule add coarsegrid;

out = struct();

% ------------------------------------------------ case A: uniform Cartesian
G  = computeGeometry(cartGrid([6, 4, 2]));
p  = partitionUI(G, [3, 2, 1]);
CG = coarsenGeometry(generateCoarseGrid(G, p));
out.A = packGeom(CG);
out.A.p = p;
out.A.sgn = fineToCoarseSign(CG);

rng_flux = sin(1 : G.faces.num).' + 0.25 * cos(3 * (1 : G.faces.num)).';
out.A.flux  = rng_flux;
out.A.cflux = coarsenFlux(CG, rng_flux);

[b2c_pos, b2c, locno] = invertPartition(p);
out.A.b2c_pos = b2c_pos;
out.A.b2c     = b2c;
out.A.locno   = locno;

% -------------------------------- case B: graded grid, so weighting matters
% Cells of unequal size, where a volume-weighted centroid and a plain mean of
% cell centroids give different answers.
G  = computeGeometry(tensorGrid([0 1 3 7 15], [0 1 3 7], [0 1 4]));
p  = partitionUI(G, [2, 2, 1]);
CG = coarsenGeometry(generateCoarseGrid(G, p));
out.B = packGeom(CG);
out.B.p = p;
out.B.sgn = fineToCoarseSign(CG);

% ------------------------------- case C: an L-shaped block, so faces fold
G  = computeGeometry(cartGrid([2, 2, 1]));
p  = [1; 2; 2; 2];
CG = coarsenGeometry(generateCoarseGrid(G, p));
out.C = packGeom(CG);
out.C.p = p;

% ------------------------------------------------------ case D: refineNearWell
G = computeGeometry(cartGrid([20, 20, 1], [20 20 1]));
pts = G.cells.centroids;
well = [10, 10, 0.5];
[pw, binAngle, binRadius, dist] = refineNearWell(pts, well, ...
   'angleBins', 4, 'radiusBins', 3, 'logBins', true);
out.D = struct('p', pw, 'binAngle', binAngle, 'binRadius', binRadius, ...
               'dist', dist, 'pts', pts, 'well', well);

[pw2, ~, binRadius2, ~] = refineNearWell(pts, well, ...
   'angleBins', 4, 'radiusBins', 3, 'logBins', false);
out.D.p_linear = pw2;
out.D.binRadius_linear = binRadius2;

[pw3, ~, ~, ~] = refineNearWell(pts, well, 'maxRadius', 5, ...
   'angleBins', 4, 'radiusBins', 3);
out.D.p_masked = pw3;

dest = fullfile(fileparts(mfilename('fullpath')), 'ref_coarsening.mat');
save(dest, '-struct', 'out', '-v7');
fprintf('wrote %s\n', dest);

function s = packGeom(CG)
   s = struct();
   s.cells_num       = CG.cells.num;
   s.cells_volumes   = CG.cells.volumes;
   s.cells_centroids = CG.cells.centroids;
   s.faces_num       = CG.faces.num;
   s.faces_areas     = CG.faces.areas;
   s.faces_centroids = CG.faces.centroids;
   s.faces_normals   = CG.faces.normals;
   s.faces_neighbors = CG.faces.neighbors;
   s.faces_connPos   = CG.faces.connPos;
   s.faces_fconn     = CG.faces.fconn;
end
