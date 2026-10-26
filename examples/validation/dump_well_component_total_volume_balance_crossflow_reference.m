% WellComponentTotalVolumeBalanceCrossflow in MRST 2024a: two layers that do
% not communicate, 100 bar apart, and a rate-controlled producer and
% injector perforating both, so each well's wellbore carries fluid from the
% high-pressure layer into the low one. Saves the first iteration's
% component fluxes and three steps of states, well solutions and Newton
% iterations.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset off
G = computeGeometry(cartGrid([3 1 2], [300 10 20]));
rock = makeRock(G, [100 100 1e-6]*milli*darcy, 0.2);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1 5]*centi*poise, 'rho', [1000 800], ...
                           'n', [2 2], 'c', [1e-9 1e-9]/barsa, 'cR', 1e-10/barsa);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
top = G.cells.centroids(:, 3) < 10;
state0 = initResSol(G, 100*barsa + 100*barsa*~top, [0.3 0.7]);
W = addWell([], G, rock, [1 4], 'type', 'rate', 'val', -1*meter^3/day, 'name', 'P', ...
            'comp_i', [1 0], 'sign', -1);
W = addWell(W, G, rock, [3 6], 'type', 'rate', 'val', 1*meter^3/day, 'name', 'I', ...
            'comp_i', [1 0], 'sign', 1);
model.FacilityModel = GenericFacilityModel(model);
model = model.validateModel(struct('W', W));
xflow = WellComponentTotalVolumeBalanceCrossflow(model);
model.FacilityModel.FacilityFlowDiscretization.ComponentTotalFlux = xflow;
% The fluxes of the first linearisation
st = model.validateState(state0);
[~, st] = model.getEquations(st, st, day, struct('W', W, 'bc', [], 'src', []), ...
                             'iteration', 1, 'resOnly', true);
st = model.initStateFunctionContainers(st);
Q = model.FacilityModel.getProp(st, 'ComponentTotalFlux');
Q0 = cell2mat(cellfun(@value, Q(:)', 'UniformOutput', false));
schedule = simpleSchedule([1 1 1]*day, 'W', W);
[ws, states, rep] = simulateScheduleAD(state0, model, schedule);
n = numel(states);
p = zeros(n, G.cells.num); sw = zeros(n, G.cells.num);
[bhp, qWs, qOs] = deal(zeros(n, 2));
for k = 1:n
    p(k, :) = states{k}.pressure'; sw(k, :) = states{k}.s(:, 1)';
    bhp(k, :) = [ws{k}.bhp]; qWs(k, :) = [ws{k}.qWs]; qOs(k, :) = [ws{k}.qOs];
end
its = rep.Iterations;
save(fullfile(here, 'ref_well_component_total_volume_balance_crossflow.mat'), '-v7', ...
     'Q0', 'p', 'sw', 'bhp', 'qWs', 'qOs', 'its');
fprintf('Wrote ref_well_component_total_volume_balance_crossflow.mat\n');
