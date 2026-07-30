%% Dump Phase 427 reference: getEquilPC, from MRST.

mrstModule add ad-core ad-props ad-blackoil

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WOG', ...
                           'mu', [1, 3, 0.1]*centi*poise, ...
                           'rho', [1000, 800, 100]);
fluid.pcOW = {@(s) 10*(1-s), @(s) 20*(1-s)};
fluid.pcOG = {@(s) 30*s, @(s) 40*s};
model = GenericBlackOilModel(G, rock, fluid, ...
                            'water', true, 'oil', true, 'gas', true);
model = model.validateModel();

cells = [1; 3];
satnum = 2;
samples = [0.1; 0.4; 0.9];
[PC, out_sign, out_scalers] = getEquilPC(model, satnum, cells);
out_curves = zeros(numel(samples), numel(PC));
for i = 1:numel(PC)
    out_curves(:, i) = PC{i}(samples);
end

save(fullfile(thisDir, 'ref_p427_equil_pc.mat'), '-v7', ...
     'cells', 'satnum', 'samples', 'out_curves', 'out_sign', 'out_scalers');
fprintf('Wrote ref_p427_equil_pc.mat to %s\n', thisDir);
