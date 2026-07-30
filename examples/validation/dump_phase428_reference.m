%% Dump Phase 428 reference: setTimeDiscretization, from MRST.

mrstModule add ad-core ad-props ad-blackoil

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([2, 1, 1], [2, 1, 1]));
rock = makeRock(G, 100*milli*darcy, 0.2);
fluid = initSimpleADIFluid('phases', 'WO');

makeModel = @() GenericBlackOilModel(G, rock, fluid, ...
                                    'water', true, 'oil', true, 'gas', false);

m = setTimeDiscretization(makeModel(), 'fim');
out_is_fim = isa(m.FlowDiscretization.getFlowStateBuilder(), ...
                 'ImplicitFlowStateBuilder');

m = setTimeDiscretization(makeModel(), 'aim', ...
                          'firstStepImplicit', false, ...
                          'saturationCFL', 0.7);
b = m.FlowDiscretization.getFlowStateBuilder();
out_aim = [isa(b, 'AdaptiveImplicitFlowStateBuilder'), ...
           b.firstStepImplicit, b.saturationCFL];

m = setTimeDiscretization(makeModel(), 'impes', ...
                          'initialStep', 12, 'changeTimestep', false);
b = m.FlowDiscretization.getFlowStateBuilder();
out_impes = [isa(b, 'ExplicitFlowStateBuilder'), ...
             b.initialStep, b.changeTimestep];

save(fullfile(thisDir, 'ref_p428_time_discretization.mat'), '-v7', ...
     'out_is_fim', 'out_aim', 'out_impes');
fprintf('Wrote ref_p428_time_discretization.mat to %s\n', thisDir);
