%% Reduced deterministic reference for blackoilSectorModelExample.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props test-suite
gravity reset; gravity off

[~, ~, fluid] = setupSPE1();
fluid.krW = @(s) s.^2;

dims = [4, 4, 1];
physicalDims = [80, 80, 8]*meter;
G = computeGeometry(cartGrid(dims, physicalDims));
rock = makeRock(G, 0.3*darcy, 0.3);
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);

p0 = 300*barsa;
s0 = [0.2, 0.8, 0];
state0 = initResSol(G, p0, s0);
state0.rs = repmat(200, G.cells.num, 1);

T_rate = 10*year;
ij = ceil(G.cartDims./2);
W = verticalWell([], G, rock, ij(1), ij(2), [], ...
    'val', -0.25*sum(model.operators.pv)/T_rate, ...
    'type', 'orat', 'comp_i', [1, 1, 1]/3, ...
    'sign', -1, 'name', 'P1');
W.lims.bhp = 100*barsa;

bc = [];
for side = {'xmin', 'xmax', 'ymin', 'ymax'}
    bc = pside(bc, G, side{1}, p0, 'sat', s0);
end
bc.dissolution = [1, 0, 0; 0, 1, 0; 0, 200, 1];

dt = rampupTimesteps(2*day, day, 1);
[wsOpen, statesOpen] = simulateScheduleAD( ...
    state0, model, simpleSchedule(dt, 'W', W, 'bc', bc));
[wsClosed, statesClosed] = simulateScheduleAD( ...
    state0, model, simpleSchedule(dt, 'W', W, 'bc', []));

n = numel(dt); nc = G.cells.num;
[pOpen, pClosed, swOpen, swClosed, sgOpen, sgClosed, rsOpen, rsClosed] = deal(zeros(n, nc));
[qOsOpen, qOsClosed, bhpOpen, bhpClosed] = deal(zeros(n, 1));
for k = 1:n
    pOpen(k, :) = statesOpen{k}.pressure;
    pClosed(k, :) = statesClosed{k}.pressure;
    swOpen(k, :) = statesOpen{k}.s(:, 1);
    swClosed(k, :) = statesClosed{k}.s(:, 1);
    sgOpen(k, :) = statesOpen{k}.s(:, 3);
    sgClosed(k, :) = statesClosed{k}.s(:, 3);
    rsOpen(k, :) = statesOpen{k}.rs;
    rsClosed(k, :) = statesClosed{k}.rs;
    qOsOpen(k) = wsOpen{k}.qOs;
    qOsClosed(k) = wsClosed{k}.qOs;
    bhpOpen(k) = wsOpen{k}.bhp;
    bhpClosed(k) = wsClosed{k}.bhp;
end
producerCell = W.cells(1) - 1;
WI = W.WI;

save(fullfile(thisDir, 'ref_ad_blackoil_sector_model.mat'), '-v7', ...
    'dims', 'physicalDims', 'dt', 'producerCell', 'WI', ...
    'pOpen', 'pClosed', 'swOpen', 'swClosed', ...
    'sgOpen', 'sgClosed', 'rsOpen', 'rsClosed', ...
    'qOsOpen', 'qOsClosed', 'bhpOpen', 'bhpClosed');
fprintf('Wrote reduced blackoilSectorModelExample reference.\n');
