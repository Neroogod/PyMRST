%% Dump Phase 42 reference: equilibrium initialisation, from MRST ad-core.
%
% Run in MATLAB after starting MRST. Writes ref_p42_equil.mat.
%
% Two cases through MRST's initStateBlackOilAD:
%
%   1. a sharp water-oil contact (no capillary pressure);
%   2. the same, with a capillary pressure - which turns the sharp contact into
%      a capillary transition zone.
%
% What is dumped is the initial state itself: the pressure MRST decides the
% model should *carry* (which is not the oil pressure everywhere - see below)
% and the saturations.

mrstModule add ad-core ad-props ad-blackoil

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

nz    = 40;
top   = 1000;
thick = 100;
woc   = 1050;

G = cartGrid([1, 1, nz], [10, 10, thick]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + top;
G = computeGeometry(G);
nc = G.cells.num;
rock = makeRock(G, 100 * milli * darcy, 0.25);

pRef  = 200 * barsa;
rhoW  = 1000;
rhoO  =  800;
pc0   = 0.5 * barsa;

for usePc = [false, true]
    fluid = initSimpleADIFluid('phases', 'WO', ...
                               'mu',   [1, 5] * centi * poise, ...
                               'rho',  [rhoW, rhoO], ...
                               'n',    [2, 2], ...
                               'c',    [0, 0], ...
                               'pRef', pRef);
    if usePc
        fluid.pcOW = @(sw) pc0 * (1 - sw);
    end

    model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                                 'oil', true, 'gas', false);
    model = model.validateModel();

    region = getInitializationRegionsBlackOil(model, woc, ...
                 'datum_depth', top, 'datum_pressure', pRef);

    [state, pressures] = initStateBlackOilAD(model, region);

    if usePc
        pc_p = state.pressure;  pc_s = state.s;  pc_pp = pressures;   %#ok<NASGU>
    else
        no_p = state.pressure;  no_s = state.s;  no_pp = pressures;   %#ok<NASGU>
    end
end

z = G.cells.centroids(:, 3);
save(fullfile(thisDir, 'ref_p42_equil.mat'), '-v7', ...
     'no_p', 'no_s', 'no_pp', 'pc_p', 'pc_s', 'pc_pp', ...
     'z', 'nz', 'top', 'thick', 'woc', 'pRef', 'rhoW', 'rhoO', 'pc0');

fprintf('Wrote ref_p42_equil.mat to %s\n', thisDir);
fprintf('  sharp contact : sW goes %.3f -> %.3f, %d cells strictly between\n', ...
        no_s(1,1), no_s(end,1), sum(no_s(:,1) > 1e-6 & no_s(:,1) < 1-1e-6));
fprintf('  with pc       : sW goes %.3f -> %.3f, %d cells strictly between\n', ...
        pc_s(1,1), pc_s(end,1), sum(pc_s(:,1) > 1e-6 & pc_s(:,1) < 1-1e-6));
