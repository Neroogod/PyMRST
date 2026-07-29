%% Dump Phase 421 reference: DG cell dimensions, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p421_cell_dims.mat.
%
% Pins PyMRST's `compute_cell_dimensions` against MRST's `computeCellDimensions`:
% the per-cell reference-frame span `dx` and bounding box `xMin`/`xMax`, on a
% Cartesian and a perturbed 2-D grid and a 3-D grid.

mrstModule add dg vem vemmech

thisDir = fileparts(mfilename('fullpath'));
out = struct();

% 2-D Cartesian.
G = computeGeometry(cartGrid([3, 4], [6, 8]));
G = computeCellDimensions(G);
out.cart2_dx = G.cells.dx; out.cart2_xmin = G.cells.xMin; out.cart2_xmax = G.cells.xMax;

% 2-D perturbed (non-uniform node positions).
G = cartGrid([4, 3], [4, 3]);
c = G.nodes.coords; L = 4;
c(:,1) = c(:,1) + 0.2*sin(pi*c(:,1)/L).*sin(pi*c(:,2)/3);
G.nodes.coords = c;
G = computeGeometry(G);
G = computeCellDimensions(G);
out.pert2_dx = G.cells.dx; out.pert2_xmin = G.cells.xMin; out.pert2_xmax = G.cells.xMax;

% 3-D Cartesian.
G = computeGeometry(cartGrid([2, 3, 2], [4, 3, 2]));
G = computeCellDimensions(G);
out.cart3_dx = G.cells.dx; out.cart3_xmin = G.cells.xMin; out.cart3_xmax = G.cells.xMax;

save(fullfile(thisDir, 'ref_p421_cell_dims.mat'), '-v7', '-struct', 'out');
fprintf('Wrote ref_p421_cell_dims.mat\n');
