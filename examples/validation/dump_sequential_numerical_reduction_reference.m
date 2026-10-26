% Reference for the 'numerical' pressure reduction of PressureModel
% (validate_ad_core_numerical_pressure_reduction_factors.py): a 1D two-phase
% waterflood on a GenericBlackOilModel (the sequential toy case), split with
% getSequentialModelFromFI, the pressure model's reductionStrategy set to
% 'numerical', three one-day steps. Per step the states and iterations.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props sequential
mrstSettings('set', 'useMEX', false);
gravity reset off
n = 10;
G = computeGeometry(cartGrid(n, n));
rock = makeRock(G, darcy, 0.3);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1e-3, 2e-3], 'rho', [1000, 800], 'n', [2, 2]);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
rate = sum(poreVolume(G, rock))/(10*day);
bc = fluxside([], G, 'xmin', rate, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', 0*barsa, 'sat', [0, 1]);
schedule = simpleSchedule(repmat(day, 3, 1), 'bc', bc);
state0 = initResSol(G, 50*barsa, [0.2, 0.8]);
seq = getSequentialModelFromFI(model);
seq.pressureModel.reductionStrategy = 'numerical';
[~, states, reports] = simulateScheduleAD(state0, seq, schedule);
p = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sw = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
its = cellfun(@(r) r.Iterations, reports.ControlstepReports);
rfclass = class(seq.validateModel().pressureModel.parentModel.PVTPropertyFunctions.getStateFunction('PressureReductionFactors'));
save(fullfile(here, 'ref_sequential_numerical_reduction.mat'), '-v7', 'p', 'sw', 'its', 'rfclass');
