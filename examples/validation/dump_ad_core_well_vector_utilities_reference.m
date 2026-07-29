%% MRST reference for padRatesAndCompi, wellSolToVector and recoverVars.

mrstModule add ad-core ad-props ad-blackoil
thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([1, 1, 1], [1, 1, 1]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
    'rho', [800, 100], 'mu', [5, 0.1]*centi*poise, 'c', [0, 0]);
model = GenericBlackOilModel(G, rock, fluid, ...
    'water', false, 'oil', true, 'gas', true);
W = addWell([], G, rock, 1, 'type', 'rate', 'val', 1, ...
    'name', 'W', 'sign', 1, 'comp_i', [0.25, 0.75]);
[q_pad, W_pad, active] = padRatesAndCompi({[2; 3], [4; 5]}, W, model);
qW = q_pad{1}; qO = q_pad{2}; qG = q_pad{3};
compi = W_pad.compi;

base = struct('sign', 1, 'qWs', 0, 'qOs', 0, 'qGs', 0, 'bhp', 0);
s1 = repmat(base, 1, 2); s2 = repmat(base, 1, 2);
s1(1) = struct('sign', 1,  'qWs', 1,   'qOs', 2,   'qGs', 3,   'bhp', 100);
s1(2) = struct('sign', -1, 'qWs', -4,  'qOs', -5,  'qGs', -6,  'bhp', 90);
s2(1) = struct('sign', 1,  'qWs', 7,   'qOs', 8,   'qGs', 9,   'bhp', 110);
s2(2) = struct('sign', -1, 'qWs', -10, 'qOs', -11, 'qGs', -12, 'bhp', 95);
[qWs, qOs, qGs, bhp] = wellSolToVector({s1, s2});

eq = struct();
eq.val = [7; -1];
eq.jac = {
    sparse([2, 0; 1, 3]), ...
    sparse([4, 1; 0, 2]), ...
    sparse([1; 2])
};
x = recoverVars(eq, 2, {[0.5; -1], 2});

save(fullfile(thisDir, 'ref_ad_core_well_vector_utilities.mat'), '-v7', ...
    'qW', 'qO', 'qG', 'compi', 'active', ...
    'qWs', 'qOs', 'qGs', 'bhp', 'x');
fprintf('Wrote ref_ad_core_well_vector_utilities.mat\n');
