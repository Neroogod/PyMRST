%% Reference data for processPINCH

thisDir = fileparts(mfilename('fullpath'));
grdecl = cartesianGrdecl(0:2, 0:2, 0:5:15);
grdecl.ACTNUM = ones(12, 1);
grdecl.ACTNUM(5:8) = 0;
grdecl.MULTZ = [0.8; 0.7; 0.6; 0.5; ...
                0.3; 0.4; 0.2; 0.1; ...
                0.9; 0.95; 0.85; 0.88];
grdecl.PINCH = {5.0, 'GAP', 0, 'TOPBOT', 'TOP'};
G = processGRDECL(grdecl);

nnc_top = processPINCH(grdecl, G);
top_cells = nnc_top.cells;
top_faces = nnc_top.faces;
top_mult = nnc_top.mult;

grdecl.PINCH{5} = 'ALL';
nnc_all = processPINCH(grdecl, G);
all_cells = nnc_all.cells;
all_faces = nnc_all.faces;
all_mult = nnc_all.mult;

grdecl.PINCH{1} = 4.99;
nnc_excluded = processPINCH(grdecl, G);
excluded_is_empty = isempty(nnc_excluded);

save(fullfile(thisDir, 'ref_grid_process_pinch.mat'), '-v7', ...
     'top_cells', 'top_faces', 'top_mult', ...
     'all_cells', 'all_faces', 'all_mult', 'excluded_is_empty');
fprintf('Wrote processPINCH reference to %s\n', thisDir);
