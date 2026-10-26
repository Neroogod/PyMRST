% immiscibleTimeIntegrationExample in MRST 2024a without its figures or the
% packed-problem output folders: fully implicit, explicit, adaptive implicit
% and explicit with CFL 5 (stepFunctionIsLinear), then the three sequential
% variants. Saves each run's water saturation per report step, iterations,
% ministeps, AIM's implicit fraction and the CFL estimate the example plots.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat ad-props sequential
L = 1000*meter;
G = cartGrid(150, L);
G = computeGeometry(G);
x = G.cells.centroids/L;
poro = repmat(0.5, G.cells.num, 1);
regs = {[0.20, 0.30, 0.05], [0.45, 0.55, 0.1], [0.70, 0.80, 0.05]};
for i = 1:numel(regs)
    R = regs{i};
    poro(x > R(1) & x < R(2)) = R(3);
end
rock = makeRock(G, 1*darcy, poro);
time = 500*day;
fluid = initSimpleADIFluid('phases', 'WO', 'n', [1 1]);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
state0 = initResSol(G, 50*barsa, [0, 1]);
pv = poreVolume(G, rock);
injRate = sum(pv)/time;
bc = fluxside([], G, 'xmin', injRate, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', 0*barsa, 'sat', [0, 1]);
n  = 150;
dT = time/n;
timesteps = rampupTimesteps(time, dT, 8);
schedule = simpleSchedule(timesteps, 'bc', bc);
model_explicit = setTimeDiscretization(model, 'explicit', 'verbose', 2);
model_aim = setTimeDiscretization(model, 'adaptive-implicit', 'verbose', 2);
model_explicit_largedt = setTimeDiscretization(model, 'explicit', 'verbose', true, ...
    'saturationCFL', 5, 'compositionCFL', inf);
model_explicit_largedt.stepFunctionIsLinear = true;
modelseq = getSequentialModelFromFI(model);
modelseq_explicit = setTimeDiscretization(modelseq, 'explicit', 'verbose', true);
models = {model, model_explicit, model_aim, model_explicit_largedt, modelseq, modelseq_explicit};
names = {'FIM', 'EXPL', 'AIM', 'EXPL_unstable', 'SI', 'SE'};
sw = cell(1, numel(models)); its = sw; mini = sw; p = sw;
for k = 1:numel(models)
    [~, states, rep] = simulateScheduleAD(state0, models{k}, schedule);
    sw{k} = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
    p{k} = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
    its{k} = rep.Iterations;
    mini{k} = getReportMinisteps(rep);
    if k == 1
        fim_states = states;
    end
    if k == 3
        imp_frac = cellfun(@(x) sum(x.implicit)/numel(x.implicit), states);
    end
end
model_cfl = model.setupStateFunctionGroupings();
cfl = zeros(numel(fim_states), G.cells.num);
for stepNo = 1:numel(fim_states)
    cfl(stepNo, :) = estimateSaturationCFL(model_cfl, fim_states{stepNo}, schedule.step.val(stepNo))';
end
save(fullfile(here, 'ref_immiscible_time_integration.mat'), 'names', 'sw', 'p', 'its', 'mini', ...
     'imp_frac', 'cfl', 'timesteps', '-v7');
