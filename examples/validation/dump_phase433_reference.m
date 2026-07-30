%% Dump Phase 433: computeSourcesAndBoundaryConditionsAD, from MRST.

mrstModule add ad-core ad-props ad-blackoil
gravity reset off
thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([2, 1, 1], [2, 1, 1]));
rock = makeRock(G, 1*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'rho', [1000, 800], ...
                           'mu', [1, 1]*centi*poise, 'n', [2, 2], ...
                           'c', [0, 0]);
model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                             'oil', true, 'gas', false);
model = model.validateModel();

left = find(G.faces.neighbors(:, 1) == 0 & G.faces.neighbors(:, 2) == 1, 1);
right = find(G.faces.neighbors(:, 1) == 2 & G.faces.neighbors(:, 2) == 0, 1);

pressure = {[100; 90], [100; 90]};
s = {[0.2; 0.4], [0.8; 0.6]};
mob = {[1; 2], [3; 2]};
rho = {[1100; 1200], [760; 720]};
dissolved = [];

forces.bc.face = [left; right];
forces.bc.type = {'pressure'; 'flux'};
forces.bc.value = [110; -8];
forces.bc.sat = [0.75, 0.25; 0.1, 0.9];
forces.src.cell = [1; 2];
forces.src.rate = [5; -4];
forces.src.sat = [1, 0; 0, 1];

[src, bc] = computeSourcesAndBoundaryConditionsAD( ...
    model, pressure, s, mob, rho, dissolved, forces);

T_all = model.operators.T_all;
cell_centroids = G.cells.centroids;
face_centroids = G.faces.centroids;
neighbors = G.faces.neighbors;
bc_faces = forces.bc.face;
src_mass = [value(src.phaseMass{1}), value(src.phaseMass{2})];
src_vol = [value(src.phaseVolume{1}), value(src.phaseVolume{2})];
src_map = full(src.mapping);
src_cells = src.sourceCells;
bc_mass = [value(bc.phaseMass{1}), value(bc.phaseMass{2})];
bc_vol = [value(bc.phaseVolume{1}), value(bc.phaseVolume{2})];
bc_map = full(bc.mapping);
bc_cells = bc.sourceCells;

save(fullfile(thisDir, 'ref_p433_sources_bc.mat'), '-v7', ...
     'T_all', 'cell_centroids', 'face_centroids', 'neighbors', 'bc_faces', ...
     'src_mass', 'src_vol', 'src_map', 'src_cells', ...
     'bc_mass', 'bc_vol', 'bc_map', 'bc_cells');
fprintf('Wrote ref_p433_sources_bc.mat to %s\n', thisDir);
