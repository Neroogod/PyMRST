function dump_ad_blackoil_spe10_model1_reference()
% Setup and first-report reference for tenthCSP_Model_I.
warning('off', 'all');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrst = fullfile(fileparts(root), 'mrst-2024a-Geomech');
cd(mrst); startup; mrstModule add ad-core ad-blackoil ad-props spe10;
gravity reset on
cartDims = [100, 1, 20];
physDims = cartDims .* [25, 25, 2.5]*ft;
G = computeGeometry(cartGrid(cartDims, physDims));
rock = getSPE10_model_1_rock();
IJK = gridLogicalIndices(G);
I = find(IJK{1}(:,1) == 1);
P = find(IJK{1}(:,1) == G.cartDims(1));
W = addWell([], G, rock, I, 'Comp_i', [0, 1], 'Type', 'rate', ...
    'Val', 6.97*meter^3/day, 'Radius', 0.5*ft, 'Dir', 'z', ...
    'Sign', +1, 'Name', 'I', 'refDepth', 0*ft);
W = addWell(W, G, rock, P, 'Comp_i', [1, 1], 'Type', 'bhp', ...
    'Val', 95*psia, 'Radius', 0.5*ft, 'Dir', 'z', ...
    'Sign', -1, 'Name', 'P', 'refDepth', 0*ft);
kr_deck = getSPE10_model_1_relperm();
fluid = initSimpleADIFluid('mu', [1, 0.01]*centi*poise, ...
    'rho', [700, 1]*kilogram/meter^3, 'cR', 6.0e-5/barsa, ...
    'phases', 'OG');
fluid_kr = assignSGOF(fluid, kr_deck.PROPS.SGOF, ...
    struct('sat', 1, 'interp1d', @interpTable));
fluid.krG = fluid_kr.krG{1}; fluid.krO = fluid_kr.krOG{1};
model = GenericBlackOilModel(G, rock, fluid, 'gravity', gravity, ...
    'disgas', false, 'vapoil', false, 'water', false, 'oil', true, 'gas', true);
region = getInitializationRegionsBlackOil(model, 0, 'datum_pressure', 100*psia);
state0 = initStateBlackOilAD(model, region);
schedule = simpleSchedule(0.1*day, 'W', W);
[wellSols, states] = simulateScheduleAD(state0, model, schedule);
centroids = G.cells.centroids; perm = rock.perm; poro = rock.poro;
pressure0 = state0.pressure; pressure1 = states{1}.pressure;
sO0 = state0.s(:, 1); sG0 = state0.s(:, 2);
sO1 = states{1}.s(:, 1); sG1 = states{1}.s(:, 2);
WI_I = W(1).WI; WI_P = W(2).WI; dZ_I = W(1).dZ; dZ_P = W(2).dZ;
bhp1 = [wellSols{1}.bhp]'; qOs1 = [wellSols{1}.qOs]';
qGs1 = [wellSols{1}.qGs]';
sgof = getSPE10_model_1_relperm(); sgof = sgof.PROPS.SGOF{1};
out = fullfile(root, 'examples', 'validation', 'ref_ad_blackoil_spe10_model1.mat');
save(out, 'cartDims', 'centroids', 'perm', 'poro', 'pressure0', ...
    'pressure1', 'sO0', 'sG0', 'sO1', 'sG1', 'WI_I', 'WI_P', ...
    'dZ_I', 'dZ_P', 'bhp1', 'qOs1', 'qGs1', 'sgof', '-v7');
fprintf('Saved %s\n', out);
end
