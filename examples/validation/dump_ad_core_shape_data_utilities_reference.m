%% MRST reference for expandIfUniform/expandMatrixToCell/numelData/sizeJac.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
matrix = reshape(0:11, [3, 4])';

expanded = expandIfUniform(matrix);
e1 = expanded{1}; e2 = expanded{2}; e3 = expanded{3};
subset = expandMatrixToCell(matrix, [1, 3]);
s1 = subset{1}; s2 = subset{2}; s3 = subset{3};
single = expandIfUniform((0:3)');
n_matrix = numelData(zeros(2, 3));
n_cell = numelData({1, 2, 3});
n_scalar = numelData(4);

S = sparse(3, 7);
size_sparse = [sizeJac(S, 1), sizeJac(S, 2)];
D = DiagonalJacobian(zeros(4, 1), [4, 3], [], false);
size_diagonal = sizeJac(D);
size_diagonal_dim2 = sizeJac(D, 2);

save(fullfile(thisDir, 'ref_ad_core_shape_data_utilities.mat'), '-v7', ...
    'matrix', 'e1', 'e2', 'e3', 's1', 's2', 's3', 'single', ...
    'n_matrix', 'n_cell', 'n_scalar', 'size_sparse', ...
    'size_diagonal', 'size_diagonal_dim2');
fprintf('Wrote ref_ad_core_shape_data_utilities.mat\n');
