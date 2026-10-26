% multisegmentWellExample in MRST 2024a without its figures: SPE1 run with
% a standard and a multisegment producer over the whole schedule. Saves, per
% report step, both producers' rates and bhp, the multisegment node
% pressures and segment fluxes, the completion-cell pressures and the Newton
% iterations of both runs.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-blackoil ad-props deckformat ad-core mrst-gui test-suite

[G, rock, fluid, deck, state] = setupSPE1();
[nx, ny] = deal(G.cartDims(1),G.cartDims(2));
model = ThreePhaseBlackOilModel(G, rock, fluid, 'inputdata', deck);
model.extraStateOutput = true;
schedule = convertDeckScheduleToMRST(model, deck);
c = nx*ny + (2:7)';
prodS = addWell([], G, rock, c, 'name', 'prod', ...
                'refDepth', G.cells.centroids(1,3), ...
               'type', 'rate', 'val', -8e5*meter^3/day);
topo = [1 2 3 4 5 6 2 3  4  5  6  7
        2 3 4 5 6 7 8 9 10 11 12 13]';
cell2node = sparse((8:13)', (1:6)', 1, 13, 6);
lengths = [300*ones(6,1); nan(6,1)];
diam    = [.1*ones(6,1); nan(6,1)];
depths  = G.cells.centroids(c([1 1:end 1:end]), 3);
vols    = ones(13,1);
prodMS = convert2MSWell(prodS, 'cell2node', cell2node, 'topo', topo, 'G', G, 'vol', vols, ...
                   'nodeDepth', depths, 'segLength', lengths, 'segDiam', diam);
[wbix, vix]  = deal(1:6, 7:12);
[roughness, nozzleD, discharge, nValves] = deal(1e-4, .0025, .7, 30);
prodMS.segments.flowModel = @(v, rho, mu)...
    [wellBoreFriction(v(wbix), rho(wbix), mu(wbix), prodMS.segments.diam(wbix), ...
                      prodMS.segments.length(wbix), roughness, 'massRate'); ...
     nozzleValve(v(vix)/nValves, rho(vix), nozzleD, discharge, 'massRate')];
inj = addWell([], G, rock, 100, 'name', 'inj', 'type', 'rate', 'Comp_i', [0 0 1], ...
              'val', 2.5e6*meter^3/day,'refDepth', G.cells.centroids(1,3));
W_simple = [inj; prodS];
schedule.control.W = W_simple;
[wellSolsSimple, statesSimple, reportSimple] = simulateScheduleAD(state, model, schedule);
W = combineMSwithRegularWells(inj, prodMS);
schedule.control.W = W;
model = model.validateModel();
model.FacilityModel = model.FacilityModel.setupWells(W);
[wellSols, states, report] = simulateScheduleAD(state, model, schedule);

n = numel(wellSols);
[qMS, qSimple, qInjMS, qInjSimple] = deal(zeros(n, 3));
[bhpMS, bhpSimple, bhpInjMS, bhpInjSimple] = deal(zeros(n, 1));
[nodePressure, segmentFlux, nodeComp] = deal(zeros(n, 12), zeros(n, 12), zeros(n, 36));
[pCellsMS, pCellsSimple] = deal(zeros(n, 6));
for k = 1:n
    w = wellSols{k}(2); ws = wellSolsSimple{k}(2);
    qMS(k, :) = [w.qWs, w.qOs, w.qGs];
    qSimple(k, :) = [ws.qWs, ws.qOs, ws.qGs];
    qInjMS(k, :) = [wellSols{k}(1).qWs, wellSols{k}(1).qOs, wellSols{k}(1).qGs];
    qInjSimple(k, :) = [wellSolsSimple{k}(1).qWs, wellSolsSimple{k}(1).qOs, wellSolsSimple{k}(1).qGs];
    bhpMS(k) = w.bhp; bhpSimple(k) = ws.bhp;
    bhpInjMS(k) = wellSols{k}(1).bhp; bhpInjSimple(k) = wellSolsSimple{k}(1).bhp;
    nodePressure(k, :) = w.nodePressure';
    segmentFlux(k, :) = w.segmentFlux';
    nodeComp(k, :) = w.nodeComp(:)';
    pCellsMS(k, :) = states{k}.pressure(c)';
    pCellsSimple(k, :) = statesSimple{k}.pressure(c)';
end
itMS = cellfun(@(r) r.Iterations, report.ControlstepReports);
itSimple = cellfun(@(r) r.Iterations, reportSimple.ControlstepReports);
reservoirTime = report.ReservoirTime;
sgMS = states{end}.s(:, 3); pMS = states{end}.pressure;
sgSimple = statesSimple{end}.s(:, 3); pSimple = statesSimple{end}.pressure;
save(fullfile(here, 'ref_multisegment_well_example.mat'), '-v7', 'qMS', 'qSimple', ...
     'qInjMS', 'qInjSimple', 'bhpMS', 'bhpSimple', 'bhpInjMS', 'bhpInjSimple', ...
     'nodePressure', 'segmentFlux', 'nodeComp', 'pCellsMS', 'pCellsSimple', 'itMS', ...
     'itSimple', 'reservoirTime', 'sgMS', 'pMS', 'sgSimple', 'pSimple');
fprintf('Wrote ref_multisegment_well_example.mat (%d steps)\n', n);
