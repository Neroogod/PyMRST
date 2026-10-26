% The first linearised SPE9 system in MRST 2024a: the equations of
% initEclipseProblemAD's model ('useMex' false, 'rowMajorAD' true) at the
% initial state for a one-day step, assembled. Saves each equation's
% residual, the matrix A, the right-hand side b (MRST's b = -r) and A\b.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-blackoil ad-core ad-props deckformat
fn = fullfile(getDatasetPath('spe9'), 'BENCH_SPE9.DATA');
[state0, model, schedule, nls] = initEclipseProblemAD(fn, 'useMex', false, 'rowMajorAD', true);
dt = 1*day;
[~, forces] = model.getDrivingForces(schedule.control(1));
model = model.validateModel(forces);
state = model.validateState(state0);
[model, state] = model.updateForChangedControls(state, forces);
[model, state] = model.prepareReportstep(state, state, dt, forces);
[model, st] = model.prepareTimestep(state, state, dt, forces);
problem = model.getEquations(state, st, dt, forces, 'iteration', 1);
problem = problem.assembleSystem();
[A, b] = problem.getLinearSystem();
names = problem.equationNames;
vals = cellfun(@(e) double(e.val), problem.equations, 'UniformOutput', false);
dx = A\b;
save(fullfile(here, 'ref_spe9_first_system.mat'), 'A', 'b', 'names', 'vals', 'dx', '-v7');
