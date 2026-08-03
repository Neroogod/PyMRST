function dump_pedfm_projection_reference(output_path)
% Direct MRST fixture for a centered, axis-aligned pMatFracNNCs3D case.

this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(fileparts(this_file))));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add hfm shale

G = computeGeometry(cartGrid([3, 3, 3], [3, 3, 3]));
G.rock = makeShaleRock(G, 1e-15, 0.1);
fracplanes.points = [1.5, 1.15, 1.15; 1.5, 1.85, 1.15; ...
                     1.5, 1.85, 1.85; 1.5, 1.15, 1.85];
fracplanes.aperture = 0.01;
fracplanes.poro = 1.0;
fracplanes.perm = 1e-10;
tol = 1e-8;
[G, fracplanes] = EDFMshalegrid(G, fracplanes, ...
    'Tolerance', tol, 'plotgrid', false, 'fracturelist', 1);
G = fracturematrixShaleNNC3D(G, tol);
normal_count = size(G.nnc.cells, 1);
normal_frac_cell = G.nnc.cells(1, 2);
wells.points = [0, 1.5, 1.5; 3, 1.5, 1.5];
wells.radius = 0.05;
[G, wells] = wellfractureNNCs3D(G, fracplanes, wells, tol);
fracture_wi = computeWIShale(G, 0.05, [normal_frac_cell, 1], 're', 0.2);
W = addWellShale([], G, G.rock, [normal_frac_cell, 1], ...
    'Radius', 0.05, 'Type', 'rate', 'Val', -1, 'Name', 'P1', ...
    'Comp_i', [0.9, 0.1], 'Sign', 1, 'refDepth', 0, ...
    'vfp_index', 7, 'c', [4, 5]);
well_cells_one_based = W.cells;
well_WI = W.WI;
well_rR = W.rR;
well_dZ = W.dZ;
well_sign = W.sign;
well_vfp_index = W.vfp_index;
well_c = W.c;
well_intersection_points = wells.WellXFracPts;
well_intersection_plane_one_based = wells.WellXFracID;
well_intersection_cell_one_based = wells.XFracCellIDs(:, 1);
G = pMatFracNNCs3D(G, tol);

projected_mf_one_based = G.nnc.cells(normal_count + 1:end, :);
projected_mm_one_based = G.nnc.pMMneighs;
projected_T = G.nnc.T(normal_count + 1:end);
projected_area = G.nnc.area(normal_count + 1:end);
projected_direction = G.nnc.pDir;
save(output_path, 'projected_mf_one_based', 'projected_mm_one_based', ...
    'projected_T', 'projected_area', 'projected_direction', 'fracture_wi', ...
    'well_intersection_points', 'well_intersection_plane_one_based', ...
    'well_intersection_cell_one_based', 'well_cells_one_based', 'well_WI', ...
    'well_rR', 'well_dZ', 'well_sign', 'well_vfp_index', 'well_c', '-v7');
fprintf('Wrote pEDFM projection fixture to %s\n', output_path);
end
