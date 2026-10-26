% eggExample's first three report steps in MRST 2024a with a direct linear
% solver (BackslashSolverAD) in place of GMRES-ILU: the run that separates the
% model from the iterative solver. Saves well curves, iterations and states.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat test-suite
[G, rock, fluid, deck] = setupEGG('realization', 0);
[state, model, schedule, nonlinear] = initEclipseProblemAD(deck, 'G', G, 'TimestepStrategy', 'none');
nonlinear.LinearSolver = BackslashSolverAD();
schedule.step.val = schedule.step.val(1:3);
schedule.step.control = schedule.step.control(1:3);
[wellSols, states, report] = simulateScheduleAD(state, model, schedule, 'NonLinearSolver', nonlinear);
W = schedule.control(1).W;
inj = find(vertcat(W.sign) > 0); prod = find(vertcat(W.sign) < 0);
bhp = getWellOutput(wellSols, 'bhp', inj);
orat = getWellOutput(wellSols, 'qOs', prod);
wrat = getWellOutput(wellSols, 'qWs', prod);
its = report.Iterations;
p = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sw = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
save(fullfile(here, 'ref_egg_example_direct.mat'), 'bhp', 'orat', 'wrat', 'its', 'p', 'sw', '-v7');
fprintf('Wrote ref_egg_example_direct.mat\n');
