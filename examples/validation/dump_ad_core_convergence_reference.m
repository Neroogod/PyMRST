% The convergence utilities in MRST 2024a on the first Newton system of a small
% two-well oil-water GenericBlackOilModel, for
% validate_ad_core_convergence.py: CNV_MBConvergence, getConvergenceValuesWells
% and printConvergenceReport (its text, captured with evalc).
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset on
G = computeGeometry(cartGrid([4, 3, 2], [400, 300, 20]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 4]*centi*poise, 'rho', [1000, 800], ...
                           'n', [2, 2], 'c', [1e-9, 1e-8]/barsa);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
W = addWell([], G, rock, [1; 13], 'type', 'rate', 'val', 50/day, 'comp_i', [1, 0], 'name', 'I');
W = addWell(W, G, rock, [12; 24], 'type', 'bhp', 'val', 150*barsa, 'comp_i', [1, 0], 'name', 'P');
state0 = initResSol(G, 200*barsa, [0.2, 0.8]);
state0.wellSol = initWellSolAD(W, model, state0);
dt = 10*day;
forces = model.getValidDrivingForces(); forces.W = W;
model = model.validateModel(forces);
state = model.validateState(state0);
[model, state] = model.prepareTimestep(state, state0, dt, forces);
problem = model.getEquations(state0, state, dt, forces, 'iteration', 1);
[A, b] = problem.getLinearSystem();
info = struct('names', {problem.equationNames}, 'types', {problem.types}, ...
              'vars', {problem.primaryVariables}, 'norm2', problem.norm(), 'normInf', problem.norm(inf));
perm = numel(problem.equations):-1:1;
[Ar, br] = problem.reorderEquations(perm).getLinearSystem();
wellName = problem.equationNames{find(~problem.indexOfType('cell'), 1)};
[pe, elim] = problem.eliminateVariable(wellName);
[Ae, be] = pe.getLinearSystem();
% CNV_MBConvergence reads FlowPropertyFunctions.getProperty, which MRST 2024a's
% groupings no longer have: record the error it stops with.
try
    CNV_MBConvergence(model, problem);
    cnv_error = '';
catch err
    cnv_error = err.message;
end
[values, names, tolerances] = getConvergenceValuesCNV(model, problem);
converged = values <= tolerances;
[wv, wn, wt, we] = getConvergenceValuesWells(model.FacilityModel, problem);
wells = struct('values', wv, 'names', {wn}, 'tolerances', wt, 'evaluated', we);
allnames = [names, wn];
txt = evalc(['printConvergenceReport(allnames, [values, wv], [converged, wv <= wt], 1); ', ...
             'printConvergenceReport(allnames, [values/10, wv/100], true(1, numel(allnames)), 2); ', ...
             'printConvergenceReport(allnames, [Inf, NaN, values(3:end), wv], false(1, numel(allnames)), 3, true);']);
save(fullfile(here, 'ref_ad_core_convergence.mat'), '-v7', 'cnv_error', 'wells', 'txt', ...
     'names', 'values', 'converged');
fprintf('%s', txt);
