%% Dump Phase 355 reference: CNV / material-balance convergence, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p355_convergence.mat.
%
% PyMRST used to compute the CNV (max-norm) and MB (material-balance) convergence
% values with a copy of the same arithmetic inside every model class. AC19
% factored that into `ad_core/utils/convergence.py`, matching MRST's
% `getConvergenceValuesCNV`. This pins the factored formula against MRST number
% for number.
%
% The residual is a *real* Newton step's, not a contrived one: SPE 1 is run for
% one report step, and the linearized problem is re-assembled at that state.
% That residual is non-trivial in every phase - including gas, whose `b ~ 262`
% is the whole reason MRST weights each phase by `B_avg = mean(1/b)`.
%
% What is saved is MRST's own `getConvergenceValues` output - the CNV_* and MB_*
% values it would print - together with the `b`, `pv` and `dt` that produced
% them, so PyMRST can either compare the values directly or recompute from the
% same inputs.

mrstModule add ad-core ad-props ad-blackoil deckformat

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

% -- SPE 1, exactly as validate_phase5b builds it ----------------------------
deck = convertDeckUnits(readEclipseDeck( ...
    fullfile(thisDir, '..', '..', '..', 'mrst-2024a-Geomech', ...
             'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA')));
try; mrstSettings('set', 'useMEX', false); catch; end
[state0, model, schedule] = initEclipseProblemAD(deck, 'useMex', false, ...
                                                 'AutoDiffBackend', 'sparse');
model = model.validateModel();

% The convergence values a solver sees at the *first* iteration of step 1: the
% problem assembled at the initial state, wells active. State equals state0, so
% the accumulation term is zero and what is left is the well-driven residual -
% non-trivial in every phase, which is exactly what makes it a real test.
dt = schedule.step.val(1);
forces = model.getValidDrivingForces();
forces.W = schedule.control(1).W;

state0 = model.validateState(state0);
problem = model.getEquations(state0, state0, dt, forces, 'iteration', 1, ...
                             'resOnly', false);
[values, tolerances, names] = model.getConvergenceValues(problem);

% the ingredients, so PyMRST can recompute CNV/MB from *identical* inputs -
% which isolates the factored formula from any difference in how the two codes
% initialise the well bhp (and hence the iteration-1 residual).
b  = model.getProps(state0, 'ShrinkageFactors');
pv = value(model.getProps(state0, 'PoreVolume'));
b_mat = zeros(numel(pv), numel(b));
for i = 1:numel(b)
    B = value(b{i});
    b_mat(:, i) = B .* ones(numel(pv), 1);
end
rhoS = model.getSurfaceDensities();

% the per-phase cell residuals MRST's criterion actually saw, in surface volume
% (a GenericReservoirModel assembles in mass, so divide the surface density back
% out - exactly what getConvergenceValuesCNV does with its `isMass` branch).
[~, phaseNames] = model.getPhaseNames();     % {'water','oil','gas'}
res_surf = zeros(numel(pv), numel(b));
for ph = 1:numel(b)
    eq_ix = problem.indexOfEquationName(phaseNames{ph});
    eq = value(problem.equations{eq_ix});
    res_surf(:, ph) = eq ./ value(rhoS(ph));
end

names = char(names);      % cellstr -> char matrix for -v7 save

save(fullfile(thisDir, 'ref_p355_convergence.mat'), '-v7', ...
     'values', 'tolerances', 'names', 'b_mat', 'pv', 'dt', 'rhoS', 'res_surf');

fprintf('Wrote ref_p355_convergence.mat to %s\n', thisDir);
fprintf('  dt = %.4g d\n', dt/day);
for i = 1:numel(values)
    fprintf('  %-8s = %.6e  (tol %.1e)\n', strtrim(names(i,:)), ...
            values(i), tolerances(i));
end
