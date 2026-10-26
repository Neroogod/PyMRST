% Reference for demoPackedProblems (validate_ad_core_demo_packed_problems.py):
% the script up to its figure - both packed problems simulated with
% simulatePackedProblem and read back with getPackedSimulatorOutput and
% getMultiplePackedSimulatorOutputs - with old output cleared first. The
% background runs that follow in the script are not repeated here.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
mrstSettings('set', 'useMEX', false);
gravity reset off
G = cartGrid([100, 1], [1000, 1000]);
rock = makeRock(G, 0.1*darcy, 0.3);
G = computeGeometry(G);
fluid_1 = initSimpleADIFluid('mu', [1, 1]*centi*poise, 'n', [1, 1], 'phases', 'wo', 'rho', [1000, 500]);
model_1 = TwoPhaseOilWaterModel(G, rock, fluid_1);
time = 10*year;
irate = sum(model_1.operators.pv)/time;
W = [];
W = addWell(W, G, rock, 1, 'type', 'rate', 'val', irate, 'comp_i', [1, 0]);
W = addWell(W, G, rock, G.cells.num, 'type', 'bhp', 'val', 100*barsa, 'comp_i', [1, 0]);
dt = rampupTimesteps(time, time/100);
schedule = simpleSchedule(dt, 'W', W);
state0 = initResSol(G, 150*barsa, [0, 1]);
fluid_2 = initSimpleADIFluid('mu', [1, 1]*centi*poise, 'n', [2, 2], 'phases', 'wo', 'rho', [1000, 500]);
model_2 = TwoPhaseOilWaterModel(G, rock, fluid_2);
BaseName = 'test_packed';
problem_1 = packSimulationProblem(state0, model_1, schedule, BaseName, ...
                                'Name', 'linear_relperm', ...
                                'Description', '1D displacement with linear flux');
problem_2 = packSimulationProblem(state0, model_2, schedule, BaseName, ...
                                'Name', 'quadratic_relperm', ...
                                'Description', '1D displacement with nonlinear flux');
problems = {problem_1, problem_2};
clearPackedSimulatorOutput(problems, 'prompt', false);
[ok, status] = simulatePackedProblem(problems);
[all_ws, all_states, all_reports, names, report_time] = getMultiplePackedSimulatorOutputs(problems);
out = struct('ok', ok, 'names', {names}, 'report_time', {report_time}, 'dt', dt);
for i = 1:2
    out.(sprintf('qWs%d', i)) = cell2mat(cellfun(@(x) [x.qWs], all_ws{i}, 'UniformOutput', false));
    out.(sprintf('qOs%d', i)) = cell2mat(cellfun(@(x) [x.qOs], all_ws{i}, 'UniformOutput', false));
    out.(sprintf('bhp%d', i)) = cell2mat(cellfun(@(x) [x.bhp], all_ws{i}, 'UniformOutput', false));
    out.(sprintf('sw%d', i)) = cell2mat(cellfun(@(x) x.s(:, 1)', all_states{i}, 'UniformOutput', false));
    out.(sprintf('its%d', i)) = cellfun(@(x) x.Iterations, all_reports{i});
end
clearPackedSimulatorOutput(problems, 'prompt', false);
save(fullfile(here, 'ref_demo_packed_problems.mat'), '-struct', 'out', '-v7');
