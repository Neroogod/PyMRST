%% References for getWellPolymer and getPolymerShearMultiplier

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil

W(1) = struct('sign', 1, 'cells', [1;2], 'cp', 2.5, 'compi', [1,0,0]);
W(2) = struct('sign', -1, 'cells', 3, 'cp', [], 'compi', [0,1,0]);
W(3) = struct('sign', 1, 'cells', [4;5;6], 'cp', [], 'compi', [0,0,1]);
[wPoly, wciPoly, iInxW] = getWellPolymer(W);
iInxW0 = iInxW-1;

model.fluid.plyshearMult = @(velocity) 1 + 0.5.*velocity;
VW0 = [-0.1; 0.25; 0.5];
muWmult = [2; 3; 1.5];
[mult, VW, report] = getPolymerShearMultiplier(model, VW0, muWmult);
iterations = report.Iterations;
residual = report.Residual;
converged = report.Converged;

save(fullfile(thisDir, 'ref_ad_blackoil_polymer_utils.mat'), '-v7', ...
     'wPoly','wciPoly','iInxW0','VW0','muWmult','mult','VW', ...
     'iterations','residual','converged');
fprintf('Wrote ad-blackoil polymer utility reference to %s\n', thisDir);
