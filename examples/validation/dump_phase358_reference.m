%% Dump Phase 358 reference: ad-core numeric utilities (AC22), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p358_numeric.mat.
%
% AC22 ports the genuinely-numeric ad-core/utils helpers one file at a time.
% This pins the ones with non-trivial arithmetic against MRST's own output so
% the PyMRST ports match number for number:
%   interpolateIDW, fastInterpTable, phaseDensitiesTobfactor, mergeOrderedArrays.

mrstModule add ad-core

thisDir = fileparts(mfilename('fullpath'));

% -- interpolateIDW: scattered 2D samples, a few query points ----------------
idw_x  = [0 0; 1 0; 0 1; 1 1; 0.5 0.5];
idw_f  = [10; 20; 30; 40; 100];
idw_xq = [0.25 0.25; 0.5 0.5; 0.9 0.1; 0.5 0.0];
idw_order = 2;
idw_v = interpolateIDW(idw_x, idw_f, idw_xq, idw_order);

% -- fastInterpTable: linear with linear extrapolation -----------------------
fit_X  = [0; 1; 2; 3; 5];
fit_Y  = [0; 10; 20; 30; 60];
fit_xi = [-1; 0.5; 1.5; 4; 6];
fit_yi = fastInterpTable(fit_X, fit_Y, fit_xi);

% -- phaseDensitiesTobfactor: dead (matrix) and live (cell w/ dissolved) -----
% dead: b(:,i) = rho(:,i) / rhoS(i)
pdb_rho  = [1000 800; 1010 790];      % 2 cells x 2 phases
pdb_rhoS = [1000, 850];
pdb_b_dead = phaseDensitiesTobfactor(pdb_rho, pdb_rhoS, []);

% live: 3 phases, gas dissolved in oil (rs). Cell-array form.
rhoW = [1000; 1000]; rhoO = [850; 860]; rhoG = [1.2; 1.25];
rho_cell = {rhoW, rhoO, rhoG};
rhoS3 = [1000, 800, 1.0];
rs = [120; 110];
% dissolved{j}{i} = surface phase j per surface volume of phase i
dissolved = {{[],[],[]}, {[],[],[]}, {[], rs, []}};   % gas(3) in oil(2)
pdb_b_live = phaseDensitiesTobfactor(rho_cell, rhoS3, dissolved);
pdb_b_live = [pdb_b_live{1}, pdb_b_live{2}, pdb_b_live{3}];

% -- mergeOrderedArrays: the supported "insertion" cases ---------------------
merge1 = mergeOrderedArrays([1;2;3;4], [1;2;3;4]);
merge2 = mergeOrderedArrays([1;2;3],   [1;2;4;3]);
merge3 = mergeOrderedArrays([10;20;30],[10;15;20;30]);

save(fullfile(thisDir, 'ref_p358_numeric.mat'), '-v7', ...
     'idw_x', 'idw_f', 'idw_xq', 'idw_order', 'idw_v', ...
     'fit_X', 'fit_Y', 'fit_xi', 'fit_yi', ...
     'pdb_rho', 'pdb_rhoS', 'pdb_b_dead', 'pdb_b_live', ...
     'merge1', 'merge2', 'merge3');

fprintf('Wrote ref_p358_numeric.mat to %s\n', thisDir);
fprintf('  idw v(1) = %.6f,  fit yi(1) = %.6f (extrapolated)\n', ...
        idw_v(1), fit_yi(1));
