%% Reference data for the vemmech test-grid fixtures

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

%% ENu2LMu_3D: closed-form Lame parameters
enu_E  = [1; 2; 5; 0.3e9];
enu_nu = [0.25; 0.3; 0.2; 0.4];
[enu_lambda, enu_mu] = ENu2LMu_3D(enu_E, enu_nu);

%% linearDisplacement: the analytic fields, sampled on fixed points
ld_points2 = [0.0 0.0; 1.0 0.0; 0.0 1.0; 0.3 0.7; -0.5 0.25];
ld_points3 = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 0.2 0.5 0.9];
[sol2, names2] = linearDisplacement(2);
[sol3, names3] = linearDisplacement(3);
ld_names2 = names2;
ld_names3 = names3;
ld_values2 = [];
for i = 1:numel(sol2)
    ld_values2 = [ld_values2; sol2{i}(ld_points2)];  %#ok
end
ld_values3 = [];
for i = 1:numel(sol3)
    ld_values3 = [ld_values3; sol3{i}(ld_points3)];  %#ok
end

%% squareGrid, cartgrid: deterministic, so it compares exactly
sg_cart = squareGrid([4 3], [2 1.5], 'grid_type', 'cartgrid');
sg_cart = computeGeometry(sg_cart);
sg_cart_cells  = sg_cart.cells.num;
sg_cart_faces  = sg_cart.faces.num;
sg_cart_nodes  = sg_cart.nodes.num;
sg_cart_coords = sg_cart.nodes.coords;
sg_cart_vols   = sg_cart.cells.volumes;

sg_cart3 = computeGeometry(squareGrid([3 3 2], [3 3 2], 'grid_type', 'cartgrid'));
sg_cart3_cells = sg_cart3.cells.num;
sg_cart3_vol   = sum(sg_cart3.cells.volumes);

%% squareGrid, other types: ordering may differ, so record invariants
sg_types = {'triangle', 'pebi'};
sg_type_cells = zeros(numel(sg_types), 1);
sg_type_area  = zeros(numel(sg_types), 1);
for i = 1:numel(sg_types)
    Gt = computeGeometry(squareGrid([4 4], [1 1], 'grid_type', sg_types{i}));
    sg_type_cells(i) = Gt.cells.num;
    sg_type_area(i)  = sum(Gt.cells.volumes);
end

%% squareTest on a Cartesian grid, untriangulated
[Gs, bcs, tcs] = squareTest('cartDims', [4 4], 'L', [1 1], ...
                            'grid_type', 'cartgrid', 'triangulate', false, ...
                            'E', 1e6, 'nu', 0.25);
st_cells = Gs.cells.num;
st_nodes = Gs.nodes.num;
st_num_bc = numel(bcs);
st_side_faces = zeros(numel(bcs), 1);
st_side_nodes = zeros(numel(bcs), 1);
for i = 1:numel(bcs)
    st_side_faces(i) = numel(bcs{i}.face);
    st_side_nodes(i) = numel(bcs{i}.el_bc.disp_bc.nodes);
end
st_num_cases = numel(tcs);
st_case_names = cell(numel(tcs), 1);
for i = 1:numel(tcs)
    st_case_names{i} = tcs{i}.name;
end
% solve the first patch case and the first named case
st_patch_u = [];
st_named_u = [];
for i = 1:numel(tcs)
    if strcmp(tcs{i}.name, 'lin_disp_1')
        st_patch_u = VEM_linElast(Gs, tcs{i}.C, tcs{i}.el_bc, tcs{i}.load);
    end
    if strcmp(tcs{i}.name, 'original_2D_test')
        st_named_u = VEM_linElast(Gs, tcs{i}.C, tcs{i}.el_bc, tcs{i}.load);
    end
end
st_coords = Gs.nodes.coords;

%% squareLayersTest: layer assignment and the resulting stiffness
[Gl, ~, tcl] = squareLayersTest('cartDims', [4 4], 'L', [1 1], ...
                                'E', [1e6; 5e6], 'nu', [0.25; 0.25], ...
                                'zlayers', [0.5; 0.5]);
sl_cells = Gl.cells.num;
sl_centroids = Gl.cells.centroids;
sl_C11 = tcl{1}.C(:, 1);
sl_num_cases = numel(tcl);

%% makeCompactionTest: the boundary masks and the load
Gc = createAugmentedGrid(computeGeometry(cartGrid([3 3 3], [1 1 1])));
optc = struct('gravity_load', true, 'top_load', true, 'free_top', true, ...
              'hanging', false, 'islinear', false);
[el_bc_c, load_c] = makeCompactionTest(Gc, optc);
mc_num_nodes  = numel(el_bc_c.disp_bc.nodes);
mc_unique     = numel(unique(el_bc_c.disp_bc.nodes));
mc_mask_sum   = full(sum(el_bc_c.disp_bc.mask, 1));
mc_force_n    = 0;
mc_force_val  = 0;
if ~isempty(el_bc_c.force_bc)
    mc_force_n   = numel(el_bc_c.force_bc.faces);
    mc_force_val = el_bc_c.force_bc.force(1, :);
end
mc_load = load_c(Gc.cells.centroids);
mc_load_row = mc_load(1, :);

% and the solve it drives
Ec  = repmat(1e9, Gc.cells.num, 1);
nuc = repmat(0.25, Gc.cells.num, 1);
Cc  = Enu2C(Ec, nuc, Gc);
mc_u = VEM_linElast(Gc, Cc, el_bc_c, load_c);
mc_coords = Gc.nodes.coords;

%% complex3DGrid, box case
optb = struct('cartDims', [3 3 3], 'L', [1 1 1], 'disturb', 0, ...
              'triangulate', false, 'gtol', 0, 'vertical', false);
Gb = complex3DGrid(optb, 'box');
cg_plain_cells = Gb.cells.num;
cg_plain_faces = Gb.faces.num;
Gb = computeGeometry(Gb);
cg_plain_vol = sum(Gb.cells.volumes);

optb.triangulate = true;
Gbt = computeGeometry(complex3DGrid(optb, 'box'));
cg_tri_cells = Gbt.cells.num;
cg_tri_faces = Gbt.faces.num;
cg_tri_vol   = sum(Gbt.cells.volumes);

save(fullfile(thisDir, 'ref_vemmech_fixtures.mat'), '-v7', ...
     'enu_E', 'enu_nu', 'enu_lambda', 'enu_mu', ...
     'ld_points2', 'ld_points3', 'ld_values2', 'ld_values3', ...
     'ld_names2', 'ld_names3', ...
     'sg_cart_cells', 'sg_cart_faces', 'sg_cart_nodes', 'sg_cart_coords', ...
     'sg_cart_vols', 'sg_cart3_cells', 'sg_cart3_vol', ...
     'sg_types', 'sg_type_cells', 'sg_type_area', ...
     'st_cells', 'st_nodes', 'st_num_bc', 'st_side_faces', 'st_side_nodes', ...
     'st_num_cases', 'st_case_names', 'st_patch_u', 'st_named_u', 'st_coords', ...
     'sl_cells', 'sl_centroids', 'sl_C11', 'sl_num_cases', ...
     'mc_num_nodes', 'mc_unique', 'mc_mask_sum', 'mc_force_n', ...
     'mc_force_val', 'mc_load_row', 'mc_u', 'mc_coords', ...
     'cg_plain_cells', 'cg_plain_faces', 'cg_plain_vol', ...
     'cg_tri_cells', 'cg_tri_faces', 'cg_tri_vol');
fprintf('Wrote vemmech fixture reference to %s\n', thisDir);
