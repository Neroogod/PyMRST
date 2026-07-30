%% Dump Phase 429 reference: setMPFADiscretization, from MRST.

mrstModule add ad-core ad-props ad-blackoil mpfa

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([4, 3], [4, 3]));
nc = G.cells.num;
perm = repmat([100, 35, 55]*milli*darcy, nc, 1);
rock = makeRock(G, perm, 0.2);
fluid = initSimpleADIFluid('phases', 'WO');
model = GenericBlackOilModel(G, rock, fluid, ...
                            'water', true, 'oil', true, 'gas', false);
model = setMPFADiscretization(model);

out_gradient = full(model.operators.mpfagrad);
gp = model.FlowDiscretization.getStateFunction( ...
    'GravityPotentialDifference');
out_gravity_weight = full(gp.weight);
out_neighbors = model.operators.N;

save(fullfile(thisDir, 'ref_p429_mpfa_discretization.mat'), '-v7', ...
     'out_gradient', 'out_gravity_weight', 'out_neighbors');
fprintf('Wrote ref_p429_mpfa_discretization.mat to %s\n', thisDir);
