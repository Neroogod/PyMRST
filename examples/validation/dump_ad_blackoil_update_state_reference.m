%% Reference for updateStateBlackOilGeneric variable routing and limiting

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props
G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
rock = makeRock(G, 1, 0.2);
fluid = initSimpleADIFluid('phases', 'WOG', ...
    'mu', [1, 1, 1], 'rho', [1000, 700, 2]);
fluid.bO = @(p,rs,sat) 1 + 0.*p;
fluid.muO = @(p,rs,sat) 1 + 0.*p;
fluid.bG = @(p) 1 + 0.*p;
fluid.muG = @(p) 1 + 0.*p;
fluid.rsSat = @(p) 10 + 0.01.*p;
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
model = model.validateModel(struct('W', [], 'bc', [], 'src', []));
model.dpMaxRel = 0.2;
model.dpMaxAbs = inf;
model.dsMaxAbs = 0.2;
model.drsMaxRel = inf;
model.drsMaxAbs = inf;

state.pressure = [100; 200; 300];
state.s = [0.2, 0.8, 0; 0.3, 0.5, 0.2; 0.25, 0.5, 0.25];
state.rs = [5; 12; 13];
state.rv = zeros(3, 1);
state.status = [1; 3; 3];
state.wellSol = [];
problem = LinearizedProblem({'pressure','sw','x'}, state);
dx = {[50; -60; 30], [0.3; -0.1; 0.05], [2; -0.4; 0.1]};
forces.W = [];
forces.Wells = [];
% updateStateBlackOilGeneric in MRST 2024a names obsolete model properties
% (dpMax/drsMax/dsMax) and cannot execute. The maintained model method is the
% reference for the same variable-routing, limiting, and flash algorithm.
[updated, report] = model.updateState(state, problem, dx, forces);

pressure = updated.pressure;
s = updated.s;
rs = updated.rs;
status = updated.status;
save(fullfile(thisDir, 'ref_ad_blackoil_update_state.mat'), '-v7', ...
     'pressure','s','rs','status');
fprintf('Wrote updateStateBlackOilGeneric reference to %s\n', thisDir);
