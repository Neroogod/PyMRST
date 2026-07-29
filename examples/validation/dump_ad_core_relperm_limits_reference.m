%% Direct MRST reference for getMinMaxPhaseSaturationsFromRelPerm.

mrstModule add ad-core ad-props ad-blackoil
thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([1, 1]));
rock = makeRock(G, 1, 0.2);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 1], 'rho', [1, 1]);
fluid.krW = {@(s) min(max((s - 0.2)/0.7, 0), 1), ...
             @(s) min(max((s - 0.1)/0.8, 0), 1)};
fluid.krO = {@(s) min(max((s - 0.25)/0.65, 0), 1), ...
             @(s) min(max((s - 0.15)/0.75, 0), 1)};
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
model = model.validateModel();
[s_min, s_max] = getMinMaxPhaseSaturationsFromRelPerm(model, 0.05, 2);
save(fullfile(thisDir, 'ref_ad_core_relperm_limits.mat'), '-v7', ...
     's_min', 's_max');
fprintf('Wrote ref_ad_core_relperm_limits.mat\n');
