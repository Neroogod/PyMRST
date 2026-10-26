% Reference for ensemblePackedProblemsExample
% (validate_ad_core_ensemble_packed_problems_example.py): the script's 50
% realizations (rng(0), gaussianField porosity, Carman-Kozeny permeability) of
% a 1D waterflood driven by boundary conditions, packed under one base name.
% The script simulates them in background sessions with PackedProblemManager;
% here they are simulated in this session with simulatePackedProblem, which
% gives the same states. Saved: the porosity fields, the packed names and
% descriptions, and per realization the Newton iterations, the water
% saturation at the outlet at every step and the whole field every tenth step.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
mrstSettings('set', 'useMEX', false);
gravity reset off
rng(0);
n = 50;
G = cartGrid(100, 1000*meter);
G = computeGeometry(G);
time = 5*year;
irate = sum(G.cells.volumes)*0.5/time;
poro = zeros(G.cells.num, n);
for i = 1:n
    poro(:, i) = reshape(gaussianField(G.cartDims, [0.01 0.5], [11 3], 2.5), [], 1);
end
K = poro.^3.*(1e-5)^2./(0.81*72*(1-poro).^2);
bc = [];
bc = fluxside(bc, G, 'XMin', irate, 'sat', [1, 0]);
bc = pside(bc, G, 'XMax', 100*barsa, 'sat', [1, 0]);
dt = rampupTimesteps(time, time/100);
schedule = simpleSchedule(dt, 'bc', bc);
fluid = initSimpleADIFluid('mu', [1, 5]*centi*poise, 'n', [2, 2], 'phases', 'wo', 'cR', 1e-8/barsa, 'c', [1e-5, 1e-4]/barsa);
state0 = initResSol(G, 100*barsa, [0, 1]);
baseName = 'EnsembleExample1D';
problems = cell(1, n);
for poroNo = 1:n
    k = K(:, poroNo);
    p = poro(:, poroNo);
    caseName = sprintf('Case %d', poroNo);
    rock = makeRock(G, k, p);
    model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
    description = sprintf('Average porosity %1.2f, average perm %1.2f md', mean(p), mean(k)/(milli*darcy));
    problems{poroNo} = packSimulationProblem(state0, model, schedule, baseName, 'Name', caseName, 'Description', description);
end
clearPackedSimulatorOutput(problems, 'prompt', false);
simulatePackedProblem(problems);
[ws, states, reports, names] = getMultiplePackedSimulatorOutputs(problems);
nstep = numel(schedule.step.val);
keep = [1:10:nstep, nstep];
out = struct('poro', poro, 'names', {names}, 'dt', dt, 'keep', keep);
out.descriptions = cellfun(@(x) x.Description, problems, 'UniformOutput', false);
out.its = cell2mat(cellfun(@(r) cellfun(@(x) x.Iterations, r), reports, 'UniformOutput', false));
out.outlet = cell2mat(cellfun(@(st) cellfun(@(x) x.s(end, 1), st), states, 'UniformOutput', false));
out.sw = cell2mat(cellfun(@(st) cell2mat(cellfun(@(x) x.s(:, 1), st(keep)', 'UniformOutput', false)), ...
                  states, 'UniformOutput', false));
clearPackedSimulatorOutput(problems, 'prompt', false);
save(fullfile(here, 'ref_ensemble_packed_problems_example.mat'), '-struct', 'out', '-v7');
