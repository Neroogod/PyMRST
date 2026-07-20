%% Dump Phase 7c reference: hfm 2D EDFM (conductivity index + NNC trans).
%
% A single inclined fracture (deliberately NOT grid-aligned, to avoid the
% degenerate case where the fracture lies on cell faces) embedded in a
% Cartesian matrix grid. We save MRST's conductivity indices per matrix
% cell, the NNC list and transmissibilities, and the pressure solution.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add hfm incomp
checkLineSegmentIntersect;

celldim = [30 30];
physdim = [9 9];
G = computeGeometry(cartGrid(celldim, physdim));

fl = [2.0, 3.3, 7.0, 5.7];          % inclined fracture line
[G, fracture] = processFracture2D(G, fl);
fracture.aperture = 1 / 25;

G = CIcalculator2D(G, fracture);

% Conductivity index and embedded fracture length per matrix cell.
nc = G.cells.num;
CI_cell = zeros(nc, 1);
fA_cell = zeros(nc, 1);
for i = 1 : nc
   if ~isempty(G.cells.fracture.CI{i})
      CI_cell(i) = sum(G.cells.fracture.CI{i});
      fA_cell(i) = sum(G.cells.fracture.fA{i});
   end
end

[G, F, fracture] = gridFracture2D(G, fracture, ...
                                  'min_size', 0.1, 'cell_size', 0.3);

G.rock.perm = ones(G.cells.num, 1) * darcy;
G = makeRockFrac(G, 1e4);           % fracture permeability, darcy
fluid = initSingleFluid('mu', 1 * centi * poise, 'rho', 1000);

[G, T] = defineNNCandTrans(G, F, fracture);

nnc_cells = G.nnc.cells;
nnc_CI    = G.nnc.CI;
nnc_T     = G.nnc.T;
n_matrix  = nc;
n_total   = G.cells.num;

state = initResSol(G, 5 * barsa);
bc = [];
bc = pside(bc, G, 'LEFT',  10 * barsa, 'sat', 1);
bc = pside(bc, G, 'RIGHT',  1 * barsa, 'sat', 1);
state = incompTPFA(state, G, T, fluid, 'bc', bc, 'use_trans', true);
p = state.pressure;

frac_start = F(1).cells.start;
frac_nodes = F(1).nodes.coords;

save(fullfile(thisDir, 'ref_p7c_edfm.mat'), '-v7', ...
     'CI_cell', 'fA_cell', 'nnc_cells', 'nnc_CI', 'nnc_T', ...
     'n_matrix', 'n_total', 'p', 'frac_start', 'frac_nodes', 'fl');
fprintf('Wrote Phase-7c EDFM reference: %d matrix + %d fracture cells, %d NNCs\n', ...
        n_matrix, n_total - n_matrix, size(nnc_cells, 1));
