% Deterministic MRST 2024a references for PyMRST root utilities.
repo = '/Users/Shared/PyMRST_workspace';
utils = fullfile(repo, 'mrst-2024a-Geomech', 'utils');
% Do not use genpath: utils/octave_only intentionally shadows MATLAB
% builtins and is invalid in a MATLAB session.
addpath(utils);
addpath(fullfile(utils, 'units'));

m = [1; 3; 2]; n = [2; 1; 4];
[block_i, block_j] = blockDiagIndex(m, n);

x = [0; 1; 3; 6]; y = [0; 2; 2; 8];
xi = [-2; 0; 0.5; 1; 2; 4.5; 6; 9];
dq = dinterpq1(x, y, xi);
[linear_y, linear_d] = interpTable(x, y, xi);
[spline_y, spline_d] = interpTable(x, y, xi, 'spline', true);

disc_plain = discspace1(0, 10, 9);
refine = [0, 1; 4, 3; 7, 0.5];
disc_linear = discspace1(0, 10, 17, refine, 'linear');
disc_constant = discspace1(0, 10, 17, refine, 'constant');

time_unlimited = formatTimeRange(2*day + 3*hour + 4.005);
time_limited = formatTimeRange(2*day + 3*hour + 4, 2);

run_rows_in = [1 2; 1 2; 3 4; 3 4; 3 5];
[run_rows, run_rows_n] = rlencode(run_rows_in, 1);
run_cols_in = [1 1 2 2 3; 4 4 5 5 6];
[run_cols, run_cols_n] = rlencode(run_cols_in, 2);

subset_plus_out = subsetPlus(zeros(4, 2), [1 2; 3 4], [2; 4]);
subset_minus_out = subsetMinus(reshape(1:6, 3, 2), [10; 20], [2; 5]);
numel_out = numelValue(reshape(1:6, 2, 3));
reduce_out = reduceToDouble(reshape(1:6, 2, 3));

unique_in = [3 3 3; 1 4 1; 2 2 2; 1 1 1; 2 2 2; 3 3 3];
[unique_values, unique_ia, unique_ic] = uniqueStable(unique_in);
[unique_rows, unique_rows_ia, unique_rows_ic] = uniqueStable(unique_in, 'rows');

destination = fullfile(repo, 'PyMRST', 'scripts', 'matlab', ...
                       'ref_utils_core.mat');
save(destination, 'm', 'n', 'block_i', 'block_j', 'x', 'y', 'xi', ...
     'dq', 'linear_y', 'linear_d', 'spline_y', 'spline_d', ...
     'disc_plain', 'refine', 'disc_linear', 'disc_constant', ...
     'time_unlimited', 'time_limited', 'run_rows_in', 'run_rows', ...
     'run_rows_n', 'run_cols_in', 'run_cols', 'run_cols_n', ...
     'subset_plus_out', 'subset_minus_out', 'numel_out', 'reduce_out', ...
     'unique_in', 'unique_values', 'unique_ia', 'unique_ic', ...
     'unique_rows', 'unique_rows_ia', 'unique_rows_ic', '-v7');
fprintf('wrote %s\n', destination);
