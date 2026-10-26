% eggExample in MRST 2024a without its figures: realization 0, initialised
% and run as the example does. Saves the well curves it plots, the Newton
% iterations and ministeps, and the states at every tenth report step and
% the last.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat test-suite
realization = 0;
[G, rock, fluid, deck] = setupEGG('realization', realization);
[state, model, schedule, nonlinear] = initEclipseProblemAD(deck, 'G', G, 'TimestepStrategy', 'none');
t = tic;
[wellSols, states, report] = simulateScheduleAD(state, model, schedule, 'NonLinearSolver', nonlinear);
walltime = toc(t);
W = schedule.control(1).W;
inj = find(vertcat(W.sign) > 0);
prod = find(vertcat(W.sign) < 0);
bhp = getWellOutput(wellSols, 'bhp', inj);
orat = getWellOutput(wellSols, 'qOs', prod);
wrat = getWellOutput(wellSols, 'qWs', prod);
T = convertTo(cumsum(schedule.step.val), year);
its = report.Iterations;
nmini = cellfun(@(r) numel(r.StepReports), report.ControlstepReports);
keep = unique([10:10:numel(states), numel(states)]);
p = cell2mat(cellfun(@(s) s.pressure', states(keep), 'UniformOutput', false));
sw = cell2mat(cellfun(@(s) s.s(:, 1)', states(keep), 'UniformOutput', false));
p0 = state.pressure; s0 = state.s;
lsolver = class(nonlinear.LinearSolver);
backend = class(model.AutoDiffBackend);
names = {W.name};
save(fullfile(here, 'ref_egg_example.mat'), 'bhp', 'orat', 'wrat', 'T', 'its', 'nmini', ...
     'keep', 'p', 'sw', 'p0', 's0', 'lsolver', 'backend', 'names', 'inj', 'prod', 'walltime', '-v7');
