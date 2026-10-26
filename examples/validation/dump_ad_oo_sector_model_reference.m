% blackoilSectorModelExample, as published, without the plots; saves results.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props test-suite deckformat
[~, ~, fluid, deck] = setupSPE1();
fluid.krW = @(s) s.^2;
G = cartGrid([50, 50, 1], [1000, 1000, 100]*meter);
G = computeGeometry(G);
rock = makeRock(G, 0.3*darcy, 0.3);
s0 = [0.2, 0.8, 0];
p0 = 300*barsa;
state0 = initResSol(G, p0, s0);
state0.rs = repmat(200, G.cells.num, 1);
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
T = 10*year;
ij = ceil(G.cartDims./2);
W = verticalWell([], G, rock, ij(1), ij(2), [], ...
    'val',      -0.25*sum(model.operators.pv)/T, ...
    'type',     'orat', 'comp_i',   [1, 1, 1]/3, 'sign', -1, 'name', 'Producer');
W.lims.bhp = 100*barsa;
bc = [];
sides = {'xmin', 'xmax', 'ymin', 'ymax'};
for side = 1:numel(sides)
    bc = pside(bc, G, sides{side}, p0, 'sat', s0);
end
bc.dissolution = [1, 0, 0; 0, 1, 0; 0, 200, 1];
dt = rampupTimesteps(T, 30*day);
schedule = simpleSchedule(dt, 'W', W, 'bc', bc);
[ws, states, rep] = simulateScheduleAD(state0, model, schedule);
schedule2 = simpleSchedule(dt, 'W', W, 'bc', []);
[ws_c, states_c, rep_c] = simulateScheduleAD(state0, model, schedule2);
grab = @(st, f) cell2mat(cellfun(@(s) s.(f)', st, 'UniformOutput', false));
p1 = grab(states, 'pressure'); rs1 = grab(states, 'rs');
p2 = grab(states_c, 'pressure'); rs2 = grab(states_c, 'rs');
sg1 = cell2mat(cellfun(@(s) s.s(:, 3)', states, 'UniformOutput', false));
sg2 = cell2mat(cellfun(@(s) s.s(:, 3)', states_c, 'UniformOutput', false));
sw1 = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
sw2 = cell2mat(cellfun(@(s) s.s(:, 1)', states_c, 'UniformOutput', false));
qo1 = cellfun(@(w) w.qOs, ws); qo2 = cellfun(@(w) w.qOs, ws_c);
qg1 = cellfun(@(w) w.qGs, ws); qg2 = cellfun(@(w) w.qGs, ws_c);
bhp1 = cellfun(@(w) w.bhp, ws); bhp2 = cellfun(@(w) w.bhp, ws_c);
bhpctl1 = cellfun(@(w) strcmp(w.type, 'bhp'), ws); bhpctl2 = cellfun(@(w) strcmp(w.type, 'bhp'), ws_c);
its1 = rep.Iterations; its2 = rep_c.Iterations;
save(fullfile(here, 'ref_ad_oo_sector_model.mat'), ...
     'p1', 'p2', 'rs1', 'rs2', 'sg1', 'sg2', 'sw1', 'sw2', 'qo1', 'qo2', 'qg1', 'qg2', ...
     'bhp1', 'bhp2', 'bhpctl1', 'bhpctl2', 'its1', 'its2', 'dt', '-v7');
