%% Direct MRST reference for portable diagonal-backend utilities.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

v1 = [2; 3; 4];
v2 = [5; 6; 7];
x = sparse([1, 2; 3, 4; 5, 6]);
y = sparse([6, 5; 4, 3; 2, 1]);
[product, ~, ~] = diagProductMult(v1, v2, x, y, [], []);

base = [1; 2; 3];
subset = [1; 3];
incremented = incrementSubset(base, subset, 5);

[row, col, data] = getSparseBlocks(x, 2, 4);

save(fullfile(thisDir, 'ref_ad_core_diagonal_utils.mat'), '-v7', ...
     'v1', 'v2', 'x', 'y', 'product', 'base', 'subset', 'incremented', ...
     'row', 'col', 'data');
fprintf('Wrote ref_ad_core_diagonal_utils.mat\n');
