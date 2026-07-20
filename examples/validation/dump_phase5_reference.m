%% Dump Phase 5 reference: ad-blackoil TwoPhaseOilWaterModel waterflood.
%
% Runs a deterministic 15x15 heterogeneous waterflood (rate injector +
% bhp producer, gravity off) with simulateScheduleAD and tightened
% tolerances so the converged states are sharply defined for comparison
% with PyMRST's TwoPhaseOilWaterModelAD.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset; gravity off

G     = computeGeometry(cartGrid([15, 15, 1], [300, 300, 10]));
kvals = (50 + mod(11 * (0 : G.cells.num - 1)', 200)) * milli * darcy;
rock  = makeRock(G, kvals, 0.25);

fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu' , [1, 5] * centi * poise, ...
                           'rho', [1000, 800], ...
                           'n'  , [2, 2], ...
                           'c'  , [1e-6, 1e-4] / barsa, ...
                           'pRef', 200 * barsa);
% This MRST install carries a local MICP patch (SimpleWell.m) that reads
% fluid.rhoSinj for injectors; setting it equal to rho reproduces stock
% behavior.
fluid.rhoSinj = [1000, 800];

W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', 2e-3, ...
            'Comp_i', [1, 0], 'Radius', 0.1, 'Sign', 1);
W = addWell(W, G, rock, G.cells.num, 'Type', 'bhp', 'Val', 150 * barsa, ...
            'Comp_i', [0, 1], 'Radius', 0.1, 'Sign', -1);

model = TwoPhaseOilWaterModel(G, rock, fluid);
model.toleranceCNV = 1e-8;
model.toleranceMB  = 1e-12;
model = model.validateModel(struct('W', W, 'bc', [], 'src', []));
model.FacilityModel.toleranceWellRate = 1e-12;
model.FacilityModel.toleranceWellBHP  = 1e-6;

schedule = simpleSchedule(repmat(30 * day, [8, 1]), 'W', W);
state0   = initResSol(G, 200 * barsa, [0.2, 0.8]);

[wellSols, states] = simulateScheduleAD(state0, model, schedule);

nsteps = numel(states);
nc = G.cells.num;
p_hist  = zeros(nsteps, nc);
sw_hist = zeros(nsteps, nc);
qWs = zeros(nsteps, 2); qOs = zeros(nsteps, 2); bhp = zeros(nsteps, 2);
for i = 1 : nsteps
   p_hist(i, :)  = states{i}.pressure;
   sw_hist(i, :) = states{i}.s(:, 1);
   qWs(i, :) = [wellSols{i}.qWs];
   qOs(i, :) = [wellSols{i}.qOs];
   bhp(i, :) = [wellSols{i}.bhp];
end
WI1 = W(1).WI; WI2 = W(2).WI;

save(fullfile(thisDir, 'ref_p5_ow.mat'), '-v7', ...
     'p_hist', 'sw_hist', 'qWs', 'qOs', 'bhp', 'WI1', 'WI2');
fprintf('Wrote Phase-5 reference to %s\n', thisDir);
