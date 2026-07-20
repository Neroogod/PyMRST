function dump_phase118_reference()
% Phase 118 reference: compositionalBoundaryConditionsExample setup.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add compositional ad-core ad-props

nx = 50;
ny = 50;
dims = [nx, ny, 1];
pdims = [1000, 1000, 10];
G = cartGrid(dims, pdims);
G = computeGeometry(G);
rng(0);
K = logNormLayers(G.cartDims);
K = K(G.cells.indexMap);
rock = makeRock(G, K*milli*darcy, 0.2);

names = {'Methane'  'CarbonDioxide'  'n-Decane'};
Tcrit = [190.5640 304.1282 617.7000];
Pcrit = [4599200 7377300 2103000];
Vcrit = [9.8628e-05 9.4118e-05 6.0976e-04];
acentricFactors = [0.0114 0.2239 0.4884];
molarMass = [0.0160 0.0440 0.1423];
minP = 75*barsa;
totTime = 20*year;
pv = poreVolume(G, rock);

bc = [];
bc = fluxside(bc, G, 'xmin', sum(pv)/totTime, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', minP, 'sat', [1, 0]);
bc.components = repmat([0.1, 0.9, 0], numel(bc.face), 1);

initial = [0.3000 0.1000 0.6000];
injection = [0.1, 0.9, 0];
temperature = 423.15;
timesteps = rampupTimesteps(totTime, 75*day, 12);

bc_faces = bc.face(:);
bc_values = bc.value(:);
bc_sat = bc.sat;
bc_components = bc.components;
bc_types = bc.type(:);
pv_total = sum(pv);
cells = G.cells.num;
faces = G.faces.num;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p118_compositional_bc.mat');
save(out, 'dims', 'pdims', 'cells', 'faces', 'Tcrit', 'Pcrit', ...
    'Vcrit', 'acentricFactors', 'molarMass', 'minP', 'totTime', ...
    'pv_total', 'bc_faces', 'bc_values', 'bc_sat', 'bc_components', ...
    'bc_types', 'initial', 'injection', 'temperature', 'timesteps');
fprintf('Wrote ref_p118_compositional_bc.mat to %s\n', fileparts(out));
end
