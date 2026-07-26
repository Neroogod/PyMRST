%% Dump Phase 404 reference: VEM polyhedron integration, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p404_polyhedronint.mat.
%
% Pins the PyMRST `polyhedron_int` / tetrahedron quadrature against MRST's
% `polyhedronInt` on a 2x2x2 grid: the per-cell integrals of the 3-D monomial
% basis [1, x, y, z, x^2, y^2, z^2, xy] with a precision-3 rule. Exact for these
% polynomials, so triangulation-independent and machine-precision.

mrstModule add vem vemmech

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([2, 2, 2], [2, 2, 2]));
G = computeVEMGeometry(G);

f = @(X) [ones(size(X,1),1), X(:,1), X(:,2), X(:,3), ...
          X(:,1).^2, X(:,2).^2, X(:,3).^2, X(:,1).*X(:,2)];
I = polyhedronInt(G, (1:G.cells.num)', f, 3);

save(fullfile(thisDir, 'ref_p404_polyhedronint.mat'), '-v7', 'I');

fprintf('Wrote ref_p404_polyhedronint.mat to %s\n', thisDir);
fprintf('  %d cells x %d monomials; total volume = %.4f\n', ...
        size(I,1), size(I,2), sum(I(:,1)));
