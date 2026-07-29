%% MRST reference for initEclipseProblemAD on SPE1.

mrstModule add ad-core ad-props ad-blackoil deckformat
thisDir = fileparts(mfilename('fullpath'));
deckPath = fullfile(thisDir, '..', '..', '..', 'mrst-2024a-Geomech', ...
    'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');

[state0, model, schedule, nonlinear] = initEclipseProblemAD(deckPath);
pressure = state0.pressure;
s = state0.s;
rs = state0.rs;
step_val = schedule.step.val;
step_control = schedule.step.control;
model_class = class(model);
max_iterations = nonlinear.maxIterations;

save(fullfile(thisDir, 'ref_ad_core_init_eclipse_problem.mat'), '-v7', ...
    'pressure', 's', 'rs', 'step_val', 'step_control', ...
    'model_class', 'max_iterations');
fprintf('Wrote ref_ad_core_init_eclipse_problem.mat (%d cells, %d steps)\n', ...
    numel(pressure), numel(step_val));
