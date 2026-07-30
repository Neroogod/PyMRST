%% MRST reference for criticalPointChop/splitMatrixForReduction/reordering.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
x0 = [-2; -0.25; 2; 0.25; -2];
trial = [2; 2; -2; -2; 2];
chop_both = criticalPointChop(x0, trial, 0, 1, 'both', 'any');
chop_above_inc = criticalPointChop(x0, trial, 0, 1, 'above', 'increasing');
chop_below_dec = criticalPointChop(x0, trial, 0, 1, 'below', 'decreasing');

A = sparse(reshape(1:25, [5, 5])' + 20*eye(5));
b = reshape(0:9, [2, 5])';
sys_sparse = splitMatrixForReduction(A, b, 2, 'sparse', false);
sys_subset = splitMatrixForReduction(A, b, 2, 'subset', false);
B = full(sys_sparse.B); C = full(sys_sparse.C);
D = full(sys_sparse.D); E = full(sys_sparse.E);
f = sys_sparse.f; h = sys_sparse.h;
B_subset = full(sys_subset.B); E_subset = full(sys_subset.E);

ordering = getCellMajorReordering(3, 2);
ordering_custom = getCellMajorReordering(3, 2, ...
    'ndof', 8, 'equation_ordering', [2, 1], 'cell_ordering', [3, 1, 2]);

save(fullfile(thisDir, 'ref_ad_core_linear_algebra_utilities.mat'), '-v7', ...
    'x0', 'trial', 'chop_both', 'chop_above_inc', 'chop_below_dec', ...
    'A', 'b', 'B', 'C', 'D', 'E', 'f', 'h', 'B_subset', 'E_subset', ...
    'ordering', 'ordering_custom');
fprintf('Wrote ref_ad_core_linear_algebra_utilities.mat\n');
