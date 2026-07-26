%% Dump Phase 403 reference: VEM polygon integration, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p403_polyint.mat.
%
% Pins the PyMRST `polygon_int` / triangle quadrature against MRST's `polygonInt`
% on a 3x3 grid: the per-cell integrals of the monomial basis [1, x, y, x^2, xy,
% y^2] with a precision-3 rule. Both integrate the same polynomials exactly, so
% the result is triangulation-independent and must match to machine precision.

mrstModule add vem vemmech

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3], [3, 3]));
G = computeVEMGeometry(G);

f = @(X) [ones(size(X,1),1), X(:,1), X(:,2), X(:,1).^2, X(:,1).*X(:,2), X(:,2).^2];
I = polygonInt(G, (1:G.cells.num)', f, 3);

save(fullfile(thisDir, 'ref_p403_polyint.mat'), '-v7', 'I');

fprintf('Wrote ref_p403_polyint.mat to %s\n', thisDir);
fprintf('  %d cells x %d monomials; total area (sum col 1) = %.4f\n', ...
        size(I,1), size(I,2), sum(I(:,1)));
