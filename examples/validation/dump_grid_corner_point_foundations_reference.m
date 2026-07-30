%% Reference data for foundational MRST gridprocessing helpers

thisDir = fileparts(mfilename('fullpath'));

%% buildCornerPtPillars: sloping plus collapsed pillar, with/without scaling
grdecl.COORD = [0; 0; 0; 10; 20; 10; ...
                4; 5; 3; 4; 5; 3];
grdecl.ZCORN = [2; 8];
[pillar_x, pillar_y, pillar_z] = buildCornerPtPillars(grdecl);
[scaled_x, scaled_y, scaled_z] = buildCornerPtPillars(grdecl, 'Scale', true);

%% cart2active: preserve query order while omitting inactive cells
G.cartDims = [2, 2, 2];
G.cells.indexMap = [1; 3; 4; 7];
cart_query = [7; 2; 1; 4];
active_cells = cart2active(G, cart_query);

%% checkAndRepairZCORN: cell inversion and active-layer overlap
cell_inversion = reshape(zeros(2, 2, 2), [], 1);
tmp = reshape(cell_inversion, [2, 2, 2]);
tmp(:, :, 1) = 5;
tmp(:, :, 2) = 3;
cell_inversion = tmp(:);
repaired_cell = checkAndRepairZCORN(cell_inversion, [1, 1, 1]);

tmp = zeros(2, 2, 4);
tmp(:, :, 1) = 0;
tmp(:, :, 2) = 6;
tmp(:, :, 3) = 5;
tmp(:, :, 4) = 10;
layer_overlap = tmp(:);
repaired_layers = checkAndRepairZCORN(layer_overlap, [1, 1, 2]);
inactive_layers = checkAndRepairZCORN( ...
    layer_overlap, [1, 1, 2], 'Active', [1; 0]);

save(fullfile(thisDir, 'ref_grid_corner_point_foundations.mat'), '-v7', ...
     'pillar_x', 'pillar_y', 'pillar_z', ...
     'scaled_x', 'scaled_y', 'scaled_z', ...
     'cart_query', 'active_cells', ...
     'cell_inversion', 'repaired_cell', ...
     'layer_overlap', 'repaired_layers', 'inactive_layers');
fprintf('Wrote gridprocessing foundation reference to %s\n', thisDir);
