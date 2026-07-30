%% Full 13-node SPE1 multisegment-well reference (first report step).

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-blackoil ad-props deckformat ad-core test-suite
gravity reset; gravity on
[G, rock, fluid, deck, state0] = setupSPE1();
[nx, ny] = deal(G.cartDims(1), G.cartDims(2));
model = ThreePhaseBlackOilModel(G, rock, fluid, 'inputdata', deck);
schedule = convertDeckScheduleToMRST(model, deck);
schedule.step.val = schedule.step.val(1);
schedule.step.control = schedule.step.control(1);

c = nx*ny + (2:7)';
refDepth = G.cells.centroids(1, 3);
prodS = addWell([], G, rock, c, 'name', 'prod', 'refDepth', refDepth, ...
                'type', 'rate', 'val', -8e5*meter^3/day);
topo = [1 2 3 4 5 6 2 3 4 5 6 7; 2 3 4 5 6 7 8 9 10 11 12 13]';
cell2node = sparse((8:13)', (1:6)', 1, 13, 6);
lengths = [300*ones(6, 1); nan(6, 1)];
diameters = [.1*ones(6, 1); nan(6, 1)];
depths = G.cells.centroids(c([1 1:end 1:end]), 3);
prodMS = convert2MSWell(prodS, 'cell2node', cell2node, 'topo', topo, ...
    'G', G, 'vol', ones(13, 1), 'nodeDepth', depths, ...
    'segLength', lengths, 'segDiam', diameters);
[wbix, vix] = deal(1:6, 7:12);
[roughness, nozzleD, discharge, nValves] = deal(1e-4, .0025, .7, 30);
prodMS.segments.flowModel = @(v, rho, mu) ...
    [wellBoreFriction(v(wbix), rho(wbix), mu(wbix), ...
       prodMS.segments.diam(wbix), prodMS.segments.length(wbix), ...
       roughness, 'massRate'); ...
     nozzleValve(v(vix)/nValves, rho(vix), nozzleD, discharge, 'massRate')];
inj = addWell([], G, rock, 100, 'name', 'inj', 'type', 'rate', ...
    'Comp_i', [0 0 1], 'val', 2.5e6*meter^3/day, 'refDepth', refDepth);

schedule.control.W = [inj; prodS];
[wsSimple, statesSimple] = simulateScheduleAD(state0, model, schedule);
W = combineMSwithRegularWells(inj, prodMS);
schedule.control.W = W;
model = model.validateModel();
model.FacilityModel = model.FacilityModel.setupWells(W);
[wsMS, statesMS] = simulateScheduleAD(state0, model, schedule);

nodePressure = [wsMS{1}(2).bhp; wsMS{1}(2).nodePressure];
segmentFlux = wsMS{1}(2).segmentFlux;
reservoirPressure = statesMS{1}.pressure;
simplePressure = statesSimple{1}.pressure;
qSimple = [wsSimple{1}(2).qWs, wsSimple{1}(2).qOs, wsSimple{1}(2).qGs];
qMS = [wsMS{1}(2).qWs, wsMS{1}(2).qOs, wsMS{1}(2).qGs];
completionCells = c - 1;
perforationNodes = (8:13)' - 1;
WI = prodS.WI;
dt = schedule.step.val;

save(fullfile(thisDir, 'ref_ad_blackoil_multisegment_spe1.mat'), '-v7', ...
    'topo', 'completionCells', 'perforationNodes', 'WI', 'dt', ...
    'nodePressure', 'segmentFlux', 'reservoirPressure', 'simplePressure', ...
    'qSimple', 'qMS');
gravity reset; gravity off
fprintf('Wrote full SPE1 multisegment-well reference.\n');
