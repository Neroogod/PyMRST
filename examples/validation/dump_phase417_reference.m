%% Dump Phase 417 reference: DG basis functions, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p417_dg_basis.mat.
%
% Pins PyMRST's `dg_basis` against MRST's `dgBasis`: for several
% (degree, dim, type) combinations it dumps the multi-indices and the values of
% every basis function and its gradient at a set of test points.

mrstModule add dg

thisDir = fileparts(mfilename('fullpath'));

% Cases as (dim, degree, type), matching MRST's dgBasis(dim, degree, type).
cases = {{2, 2, 'legendre'}, {2, 3, 'poly'}, {3, 2, 'legendre'}, {2, 2, 'chebyshev'}};
out = struct();

for cNo = 1:numel(cases)
    dim = cases{cNo}{1}; degree = cases{cNo}{2}; type = cases{cNo}{3};
    basis = dgBasis(dim, degree, type);

    rng(cNo);
    x = -1 + 2*rand(7, dim);                       % points in [-1,1]^dim
    nDof = basis.nDof;
    psiVal = zeros(size(x,1), nDof);
    gradVal = zeros(size(x,1), nDof, dim);
    for i = 1:nDof
        psiVal(:,i) = basis.psi{i}(x);
        for d = 1:dim
            gp = dx(basis.psi{i}, d);                  % single gradient component
            gradVal(:,i,d) = gp(x);
        end
    end

    pre = sprintf('c%d_', cNo);
    out.([pre 'k'])    = basis.k;
    out.([pre 'x'])    = x;
    out.([pre 'psi'])  = psiVal;
    out.([pre 'grad']) = gradVal;
    out.([pre 'ndof']) = nDof;
end

save(fullfile(thisDir, 'ref_p417_dg_basis.mat'), '-v7', '-struct', 'out');
fprintf('Wrote ref_p417_dg_basis.mat (%d cases)\n', numel(cases));
