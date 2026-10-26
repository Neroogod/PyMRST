% The equilibration functions of ad-core/initialization in MRST 2024a, for
% validate_ad_core_equilibrium.py: getInitializationRegionsBase,
% initializeEquilibriumPressures, initializeEquilibriumSaturations and
% getMinMaxPhaseSaturations, on a 60-cell column.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset on
G = computeGeometry(cartGrid([1, 1, 60], [1, 1, 60]));
rock = makeRock(G, 1e-13, 0.2);
fluid = initSimpleADIFluid('phases', 'WOG', 'mu', [1, 5, 0.2]*centi*poise, 'rho', [1000, 800, 150], ...
                           'n', [2, 2, 2]);
model = GenericBlackOilModel(G, rock, fluid);
rho = {@(p, z) 1000 + 0*p, @(p, z) 800*exp(1e-9*(p - 2e7)), @(p, z) 150*(p/2e7)};
pc = {@(s) 1e5*(1 - s).^2, [], @(s) 5e4*s};
base = getInitializationRegionsBase(model, rho, [40, 15]);
defaults = struct('datum_pressure', base.datum_pressure, 'datum_depth', base.datum_depth, ...
                  'reference_index', base.reference_index, 'pc_sign', base.pc_sign, ...
                  'pc_scale', base.pc_scale, 'contacts_pc', base.contacts_pc, 's_max', base.s_max, ...
                  's_min', base.s_min, 'ncells', numel(base.cells));
region = getInitializationRegionsBase(model, rho, [40, 15], 'datum_pressure', 200*barsa, ...
    'datum_depth', 30, 'pc_functions', pc, 'pc_sign', [1, 1, -1], 'contacts_pc', [1e4, 0], ...
    's_min', [0.1, 0, 0], 's_max', [1, 1, 0.9], 'pc_scale', ones(60, 3));
p1 = initializeEquilibriumPressures(model, region);
s1 = initializeEquilibriumSaturations(model, region, p1);
overlap = region; overlap.contacts = [30, 29];
overlap.pc_functions = {@(s) 4e5*(1 - s).^2, [], @(s) 3e5*s};
p2 = initializeEquilibriumPressures(model, overlap);
s2 = initializeEquilibriumSaturations(model, overlap, p2);
fluid2 = initSimpleADIFluid('phases', 'WO', 'mu', [1, 5]*centi*poise, 'rho', [1000, 800], 'n', [2, 2]);
model2 = GenericBlackOilModel(G, rock, fluid2, 'gas', false);
region2 = getInitializationRegionsBase(model2, {rho{1}, rho{2}}, 35, 'datum_pressure', 150*barsa, ...
    'pc_functions', {@(s) 0*s, []});
p3 = initializeEquilibriumPressures(model2, region2);
s3 = initializeEquilibriumSaturations(model2, region2, p3);
[smin1, smax1] = getMinMaxPhaseSaturations(model, [1; 1; 1], [1; 2; 3]);
fluidc = fluid; fluidc.sWcon = 0.15;
modelc = GenericBlackOilModel(G, rock, fluidc);
[smin2, smax2] = getMinMaxPhaseSaturations(modelc, [1; 1], [1; 2]);
save(fullfile(here, 'ref_ad_core_equilibrium.mat'), '-v7', 'defaults', 'p1', 's1', 'p2', 's2', ...
     'p3', 's3', 'smin1', 'smax1', 'smin2', 'smax2');
fprintf('saved: %d cells with negative oil fixed in the overlap case\n', nnz(s2(:, 2) == 0 & s2(:, 1) < 1));
