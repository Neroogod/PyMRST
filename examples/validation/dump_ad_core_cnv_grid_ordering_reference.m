%% MRST reference for CNV/MB convergence and grid SYMRCM ordering.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

model = ValidationCNVModel();
problem = ValidationCNVProblem();
[converged, values, evaluated, names] = CNV_MBConvergence(model, problem);

G = struct();
G.faces = struct('neighbors', [ ...
    0, 1; 1, 2; 2, 3; 2, 4; 4, 5; 5, 0]);
G.cells = struct('num', 5);
ordering = getGridSYMRCMOrdering(G);

save(fullfile(thisDir, 'ref_ad_core_cnv_grid_ordering.mat'), '-v7', ...
    'converged', 'values', 'evaluated', 'names', 'ordering');
fprintf('Wrote ref_ad_core_cnv_grid_ordering.mat\n');
