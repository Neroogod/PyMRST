%% MRST reference for setWellSign and assignWellValuesFromControl.

mrstModule add ad-core ad-props ad-blackoil
thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([4, 1, 1], [40, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], ...
    'rho', [1000, 800, 100], 'mu', [1, 5, 0.1]*centi*poise, ...
    'c', [0, 0, 0]);
model = GenericBlackOilModel(G, rock, fluid, ...
    'water', true, 'oil', true, 'gas', true);

W = [];
W = addWell(W, G, rock, 1, 'type', 'rate', 'val', 100, ...
    'name', 'I', 'sign', 1, 'comp_i', [0.2, 0.3, 0.5]);
W = addWell(W, G, rock, 2, 'type', 'rate', 'val', -80, ...
    'name', 'P', 'sign', -1, 'comp_i', [0, 1, 0]);
W = addWell(W, G, rock, 3, 'type', 'bhp', 'val', 150, ...
    'name', 'B', 'sign', -1, 'comp_i', [0, 1, 0]);
W = addWell(W, G, rock, 4, 'type', 'orat', 'val', -40, ...
    'name', 'O', 'sign', -1, 'comp_i', [0, 1, 0]);

base = struct('status', true, 'type', '', 'val', 0, 'sign', 0, ...
    'bhp', 0, 'qWs', 1, 'qOs', 2, 'qGs', 3);
wellSol = repmat(base, 1, numel(W));
for i = 1:numel(W)
    wellSol(i).type = W(i).type;
    wellSol(i).val = W(i).val;
    wellSol(i).sign = W(i).sign;
end
wellSol = assignWellValuesFromControl(model, wellSol, W, 1, 2, 3);
bhp = [wellSol.bhp]; qWs = [wellSol.qWs];
qOs = [wellSol.qOs]; qGs = [wellSol.qGs];

Wsign = W;
Wsign(1).type = 'rate'; Wsign(1).val = -10; Wsign(1).sign = [];
Wsign(2).type = 'rate'; Wsign(2).val =  10; Wsign(2).sign = [];
Wsign(3).type = 'bhp';  Wsign(3).val = 100; Wsign(3).sign = [];
Wsign(4).type = 'bhp';  Wsign(4).val = 100; Wsign(4).sign = -1;
Wsign = setWellSign(Wsign);
signs = [Wsign.sign];

save(fullfile(thisDir, 'ref_ad_core_well_control_assignment.mat'), '-v7', ...
    'bhp', 'qWs', 'qOs', 'qGs', 'signs');
fprintf('Wrote ref_ad_core_well_control_assignment.mat\n');
