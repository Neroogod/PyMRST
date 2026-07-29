%% Direct MRST reference for ad-core diagonal backend grid operators.
%
% Covers the numeric contracts of twoPointGradient, faceAverage,
% singlePointUpwind, and discreteDivergence. Jacobians are independently
% covered by the diagonal-vs-sparse backend reference in Phase 361.

mrstModule add ad-core

thisDir = fileparts(mfilename('fullpath'));
N = [1, 2; 2, 3; 3, 4];
cells = [1; 4; 9; 16];
flux = [2; -3; 5];
flag = logical([1; 0; 1]);

grad = twoPointGradient(N, cells, [], false);
avg = faceAverage(N, cells, false);
upwind = singlePointUpwind(flag, N, cells, false);

opt = struct();
opt.N = N;
opt.nc = 4;
opt.useMex = false;
div = discreteDivergence([], flux, opt);
div_acc = discreteDivergence(ones(4, 1), flux, opt);

save(fullfile(thisDir, 'ref_ad_core_diagonal_operators.mat'), '-v7', ...
     'N', 'cells', 'flux', 'flag', 'grad', 'avg', 'upwind', 'div', 'div_acc');
fprintf('Wrote ref_ad_core_diagonal_operators.mat\n');
