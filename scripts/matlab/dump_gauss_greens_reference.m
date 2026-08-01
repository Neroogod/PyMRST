% Reference output of MRST's volumeByGaussGreens, for validate_volume_by_gauss_greens.py
out = struct();
G = computeGeometry(cartGrid([3,2,2],[6,4,2]));
out.cart.V = volumeByGaussGreens(G);
out.cart.vol = G.cells.volumes;

G2 = computeGeometry(cartGrid([3,3],[3,3]));
out.cart2d.V = volumeByGaussGreens(G2);
out.cart2d.vol = G2.cells.volumes;

% A twisted grid: the non-planar-face case.
G3 = cartGrid([4,4,2],[4,4,2]);
c = G3.nodes.coords;
interior = all(c > 1e-9 & c < repmat([4,4,2], size(c,1), 1) - 1e-9, 2);
rng(3);
c(interior,:) = c(interior,:) + (rand(nnz(interior), 3)*0.4 - 0.2);
G3.nodes.coords = c;
G3 = computeGeometry(G3);
out.twisted.V = volumeByGaussGreens(G3);
out.twisted.vol = G3.cells.volumes;
out.twisted.coords = G3.nodes.coords;

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_gauss_greens.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_gauss_greens.mat\n');
