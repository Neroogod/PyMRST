%% Dump Phase 109 reference: MRST schedule-level multisegment well variables.
%
% This deliberately small case records the MRST-side layout for a schedule
% containing one multisegment injector and one regular producer. The Python
% validator checks that PyMRST automatically packs the same extra MS-well
% primary variables into the black-oil Newton system: pN, rW, rO, and vmix.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset; gravity off

G = computeGeometry(cartGrid([3, 1, 1], [300, 50, 20]));
rock = makeRock(G, 100*milli*darcy, 0.25);
pRef = 200*barsa;
rate = 1e-6;

fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu' , [1, 5]*centi*poise, ...
                           'rho', [1000, 800], ...
                           'n'  , [2, 2], ...
                           'c'  , [1e-6, 1e-4]/barsa, ...
                           'pRef', pRef);
fluid.rhoSinj = [1000, 800];

model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                             'gas', false);
model.toleranceCNV = 1e-8;
model.toleranceMB  = 1e-12;

inj = addWell([], G, rock, [1; 2], 'Type', 'rate', 'Val', rate, ...
              'Comp_i', [1, 0], 'sign', 1, 'Radius', 0.1, 'Name', 'MSI');

topo = [1 2; 2 3];
cell2node = sparse([2; 3], [1; 2], 1, 3, 2);
nodeDepth = [1000; 1000; 1000];
segLength = [10; 10];
segDiam = [0.2; 0.2];
vol = [10; 10; 10];
injMS = convert2MSWell(inj, 'cell2node', cell2node, 'topo', topo, ...
                       'G', G, 'vol', vol, 'nodeDepth', nodeDepth, ...
                       'segLength', segLength, 'segDiam', segDiam);
injMS.segments.flowModel = @(v, rho, mu) wellBoreFriction( ...
    v, rho, mu, injMS.segments.diam, injMS.segments.length, ...
    1e-5*ones(numel(v), 1), 'massRate');

prod = addWell([], G, rock, 3, 'Type', 'bhp', 'Val', 195*barsa, ...
               'Comp_i', [0, 1], 'sign', -1, 'Radius', 0.1, 'Name', 'PROD');

W = combineMSwithRegularWells(prod, injMS);
schedule = simpleSchedule(1*day, 'W', W);
state0 = initResSol(G, pRef, [0.2, 0.8]);

model = model.validateModel();
model.FacilityModel = model.FacilityModel.setupWells(W);

n_nodes = numel(injMS.nodes.depth);
n_segments = size(injMS.segments.topo, 1);
n_perforations = numel(injMS.cells);
n_pN = n_nodes - 1;
n_rW = n_nodes - 1;
n_rO = n_nodes - 1;
n_vmix = n_segments;
topo0 = injMS.segments.topo;
cell2node0 = full(injMS.cell2node);
cells0 = injMS.cells(:);
WI0 = injMS.WI(:);
nodePressure0 = pRef*ones(n_pN, 1);
nodeComp0 = repmat([0.280898876404494, 0.719101123595506], n_pN, 1);
segmentFlux0 = rate*ones(n_vmix, 1);

% This phase validates the schedule-level variable layout. MRST's two-phase
% MS-well solve path in this local 2024a checkout still touches qGs during
% setup, so the dump intentionally records the converted layout and initial
% extra-variable counts only. PyMRST's validator performs the actual packed
% black-oil solve smoke test.
converged = false;
p_hist = nan(G.cells.num, 1);
sw_hist = nan(G.cells.num, 1);
qWs = nan(1, 2); qOs = nan(1, 2); bhp = nan(1, 2);
nodePressure = nodePressure0;
nodeComp = nodeComp0;
segmentFlux = segmentFlux0;

save(fullfile(thisDir, 'ref_p109_mswell.mat'), '-v7', ...
     'n_nodes', 'n_segments', 'n_perforations', ...
     'n_pN', 'n_rW', 'n_rO', 'n_vmix', ...
     'topo0', 'cell2node0', 'cells0', 'WI0', ...
     'nodePressure0', 'nodeComp0', 'segmentFlux0', ...
     'converged', 'p_hist', 'sw_hist', 'qWs', 'qOs', 'bhp', ...
     'nodePressure', 'nodeComp', 'segmentFlux', 'pRef', 'rate');

fprintf('Wrote ref_p109_mswell.mat to %s\n', thisDir);
fprintf('  MS variables: pN=%d rW=%d rO=%d vmix=%d\n', ...
        n_pN, n_rW, n_rO, n_vmix);
fprintf('  MRST schedule converged: %d\n', converged);
