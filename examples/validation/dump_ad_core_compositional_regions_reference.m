%% MRST reference for getInitializationRegionsCompositional.

mrstModule add ad-core ad-props compositional
thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([4, 1, 1], [40, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
    'rho', [700, 100], 'mu', [1, 1]*centi*poise, 'c', [0, 0]);

names = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit = [190.564, 425.125, 617.7];
Pcrit = [4599200.0, 3796000.0, 2103000.0];
Vcrit = [9.86e-5, 2.55e-4, 6.098e-4];
acf = [0.0114, 0.201, 0.4884];
mw = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);
model = GenericOverallCompositionModel(G, rock, fluid, mixture, 'water', false);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model = model.validateModel();

x = [0.10, 0.25, 0.65];
y = [0.80, 0.15, 0.05];
cells = [2; 4];
contacts = 1100;
datum_depth = 1000;
datum_pressure = 8.0e6;
region = getInitializationRegionsCompositional(model, contacts, ...
    'cells', cells, 'T', 350, 'x', x, 'y', y, ...
    'datum_depth', datum_depth, 'datum_pressure', datum_pressure);

p = [7.5e6; 8.0e6];
z = [975; 1025];
rho_l = region.rho{1}(p, z);
rho_v = region.rho{2}(p, z);
reference_index = region.reference_index;
s_min = region.s_min;
s_max = region.s_max;

save(fullfile(thisDir, 'ref_ad_core_compositional_regions.mat'), '-v7', ...
    'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw', 'x', 'y', 'cells', ...
    'contacts', 'datum_depth', 'datum_pressure', 'p', 'z', ...
    'rho_l', 'rho_v', 'reference_index', 's_min', 's_max');
fprintf('Wrote ref_ad_core_compositional_regions.mat\n');
