%% Reference topology for the PyMRST plotting Phase-1 port.
%
% This records entity selection and polygon/outline topology, not pixels.

thisDir = fileparts(mfilename('fullpath'));
workspace = fileparts(fileparts(fileparts(thisDir)));
mrstRoot = fullfile(workspace, 'mrst-2024a-Geomech');
addpath(mrstRoot);
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add coarsegrid;

G2 = computeGeometry(cartGrid([2, 2], [2, 2]));
[b2_faces, b2_cells] = boundaryFaces(G2, [1; 2]);

G3 = computeGeometry(cartGrid([2, 1, 1], [2, 1, 1]));
g3_boundary = boundaryFaces(G3);
[g3_node_ids, g3_face_pos, g3_face_nodes, g3_vertices] = ...
    compactFaceTopology(G3, g3_boundary);
g3_bottom = find(all(G3.faces.neighbors == 0 | ...
                     abs(G3.faces.centroids(:, 3)) < 1e-12, 2) & ...
                 abs(G3.faces.centroids(:, 3)) < 1e-12);
g3_outline_edges = collectionOutline(G3, g3_bottom);

Gc = computeGeometry(cartGrid([4, 2, 1], [4, 2, 1]));
p = partitionCartGrid(Gc.cartDims, [2, 1, 1]);
CG = generateCoarseGrid(Gc, p);
cg_boundary = boundaryFaces(CG);
counts = CG.faces.connPos(cg_boundary + 1) - CG.faces.connPos(cg_boundary);
ix = mcolon(CG.faces.connPos(cg_boundary), ...
            CG.faces.connPos(cg_boundary + 1) - 1);
cg_fine_faces = CG.faces.fconn(ix);
cg_groups = rldecode((1:numel(cg_boundary))', counts);

Gd = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
data_cells = [2; 3];
data_subset = [10; 20];
[cell_data_faces, cell_data_owners] = boundaryFaces(Gd, data_cells);
cell_data_values = zeros(numel(cell_data_owners), 1);
cell_data_values(cell_data_owners == data_cells(1)) = data_subset(1);
cell_data_values(cell_data_owners == data_cells(2)) = data_subset(2);
[node_data_ids, ~, ~, ~] = compactFaceTopology( ...
    Gd, boundaryFaces(Gd, data_cells(1)));

G1 = computeGeometry(cartGrid([4], [8]));
line_cells = [2; 3; 4];
line_values = [5; 7; 11];
line_cell_data_xy = [G1.cells.centroids(line_cells), line_values];
line_grid_xy = [G1.cells.centroids(line_cells), ...
                ones(numel(line_cells), 1)];

Gw = computeGeometry(cartGrid([1, 1, 3], [1, 1, 3]));
well_cells = [1; 2; 3];
well_completions = Gw.cells.centroids(well_cells, :);
well_head = well_completions(1, :);
well_head(3) = min(Gw.nodes.coords(:, 3)) - 2;
well_heel = well_completions(end, :);
well_trajectory = [well_head; well_completions];
first_faces = Gw.cells.faces( ...
    Gw.cells.facePos(well_cells(1)):Gw.cells.facePos(well_cells(1)+1)-1);
well_radius = 0.25*sqrt(max(Gw.faces.areas(first_faces)));

Gf = cartGrid([3, 3, 2]);
fault_geometry.FAULTS = {
    'BOUND', 1, 1, 1, 3, 1, 2, 'X-';
    'MAIN',  2, 2, 1, 2, 1, 2, 'X+'
};
fault_geometry.MULTFLT = {'BOUND', 1; 'MAIN', 1};
[plot_faults_ref, ~] = processFaults(Gf, fault_geometry);
Gf = computeGeometry(Gf);
fault_counts = [plot_faults_ref.numf]';
fault_faces = vertcat(plot_faults_ref.faces);
fault_groups = rldecode((1:numel(plot_faults_ref))', fault_counts);
fault_label_positions = zeros(numel(plot_faults_ref), 3);
for fault_index = 1:numel(plot_faults_ref)
    fault_label_positions(fault_index, :) = mean( ...
        Gf.faces.centroids(plot_faults_ref(fault_index).faces, :), 1);
end
fault_label_positions(:, 3) = fault_label_positions(:, 3) - 2.5;

Go2 = computeGeometry(cartGrid([4, 2], [4, 2]));
outline_p2 = partitionCartGrid(Go2.cartDims, [2, 1]);
outline_labels2 = [0; outline_p2];
outline_neighbors2 = outline_labels2(Go2.faces.neighbors + 1);
outline_faces2 = find(outline_neighbors2(:, 1) ~= ...
                      outline_neighbors2(:, 2));

Go3 = computeGeometry(cartGrid([4, 2, 1], [4, 2, 1]));
outline_p3 = partitionCartGrid(Go3.cartDims, [2, 1, 1]);
outline_labels3 = [0; outline_p3];
outline_neighbors3 = outline_labels3(Go3.faces.neighbors + 1);
outline_faces3 = find( ...
    outline_neighbors3(:, 1) ~= outline_neighbors3(:, 2) & ...
    all(Go3.faces.neighbors > 0, 2));

Gbn = computeGeometry(cartGrid([6, 2, 1], [6, 2, 1]));
block_partition = partitionCartGrid(Gbn.cartDims, [3, 1, 1]);
CGbn = generateCoarseGrid(Gbn, block_partition);
block_id = 2;
coarse_neighbors = CGbn.faces.neighbors;
block_neighbors = reshape( ...
    coarse_neighbors(any(coarse_neighbors == block_id, 2), :), [], 1);
block_neighbors(block_neighbors == block_id) = [];
block_neighbors = unique(block_neighbors(block_neighbors > 0));
block_boundary_faces = boundaryFaces(Gbn, block_partition == block_id);
neighbor_face_cells = arrayfun( ...
    @(neighbor) boundaryFaces(Gbn, block_partition == neighbor), ...
    block_neighbors, 'UniformOutput', false);
neighbor_face_counts = cellfun(@numel, neighbor_face_cells);
neighbor_face_pos = cumsum([1; neighbor_face_counts(:)]);
neighbor_boundary_faces = vertcat(neighbor_face_cells{:});

Gs = computeGeometry(cartGrid([3, 2, 2], [3, 2, 2]));
slice_data = (10:10 + Gs.cells.num - 1)';
[slice_cells1, slice_faces1, slice_values1] = ...
    sliceReference(Gs, slice_data, 2, 1);
[slice_cells2, slice_faces2, slice_values2] = ...
    sliceReference(Gs, slice_data, 1, 2);
[slice_cells3, slice_faces3, slice_values3] = ...
    sliceReference(Gs, slice_data, 2, 3);

Gi = computeGeometry(cartGrid([2, 2], [2, 2]));
interp_node_data = 3*Gi.nodes.coords(:, 1) - ...
                   2*Gi.nodes.coords(:, 2) + 5;
interp_query = [Gi.cells.centroids; 3, -1];
interp_affine = interpolateData( ...
    Gi, interp_node_data, [], interp_query, ...
    'interpolation', 'linear', 'extrapolation', 'linear');
interp_cell_data = Gi.cells.centroids(:, 1) + ...
                   Gi.cells.centroids(:, 2);
interp_cells_to_nodes = interpolateData( ...
    Gi, interp_cell_data, [], 'nodes', ...
    'interpolation', 'linear', 'extrapolation', 'linear');

Gsurf = computeGeometry(cartGrid([3, 3], [3, 3]));
surface_cell_data = Gsurf.cells.centroids(:, 1) + ...
                    Gsurf.cells.centroids(:, 2);
surface_node_values = interpolateData( ...
    Gsurf, surface_cell_data, [], 'nodes', ...
    'interpolation', 'linear', 'extrapolation', 'linear');
surface_cells = [1; 4; 9];
surface_all_nodes = getSortedCellNodes(Gsurf);
surface_counts = Gsurf.cells.facePos(surface_cells + 1) - ...
                 Gsurf.cells.facePos(surface_cells);
surface_ix = mcolon(Gsurf.cells.facePos(surface_cells), ...
                    Gsurf.cells.facePos(surface_cells + 1) - 1);
surface_global_nodes = surface_all_nodes(surface_ix);
surface_node_ids = unique(surface_global_nodes, 'sorted');
surface_local = zeros(Gsurf.nodes.num, 1);
surface_local(surface_node_ids) = 1:numel(surface_node_ids);
surface_cell_nodes = surface_local(surface_global_nodes);
surface_cell_pos = cumsum([1; surface_counts]);
surface_vertices = [Gsurf.nodes.coords(surface_node_ids, :), ...
                    surface_node_values(surface_node_ids)];

Gcontour = computeGeometry(cartGrid([4, 3], [8, 3]));
contour_cell_data = 2*Gcontour.cells.centroids(:, 1) - ...
                    Gcontour.cells.centroids(:, 2);
contour_lower = min(Gcontour.nodes.coords);
contour_upper = max(Gcontour.nodes.coords);
[contour_dx, contour_dy] = cellDims( ...
    Gcontour, (1:Gcontour.cells.num)');
contour_spacing = min([contour_dx, contour_dy], [], 1);
contour_dims = ceil((contour_upper-contour_lower)./contour_spacing);
[contour_x, contour_y] = ndgrid( ...
    linspace(contour_lower(1), contour_upper(1), contour_dims(1)), ...
    linspace(contour_lower(2), contour_upper(2), contour_dims(2)));
contour_values = interpolateData( ...
    Gcontour, contour_cell_data, [], ...
    [contour_x(:), contour_y(:)], ...
    'interpolation', 'linear', 'extrapolation', 'linear');
contour_values = reshape(contour_values, contour_dims);

Gpc = computeGeometry(cartGrid([3, 2, 1], [3, 2, 1]));
plot_contour_data = Gpc.cells.centroids(:, 1) + ...
                    0.2*Gpc.cells.centroids(:, 2);
plot_contour_count = 3;
plot_contour_levels = linspace(min(plot_contour_data), ...
                               max(plot_contour_data), ...
                               plot_contour_count + 1);
plot_contour_ext = cell(plot_contour_count, 1);
plot_contour_int = cell(plot_contour_count, 1);
plot_contour_external = any(Gpc.faces.neighbors == 0, 2);
for contour_index = 1:plot_contour_count
    in_band = [false; plot_contour_data >= ...
        plot_contour_levels(contour_index) & plot_contour_data <= ...
        plot_contour_levels(contour_index + 1)];
    touches_band = any(in_band(Gpc.faces.neighbors + 1), 2);
    plot_contour_ext{contour_index} = find( ...
        touches_band & plot_contour_external);
    below = [false; plot_contour_data < ...
             plot_contour_levels(contour_index + 1)];
    above = [false; plot_contour_data > ...
             plot_contour_levels(contour_index + 1)];
    plot_contour_int{contour_index} = find( ...
        any(below(Gpc.faces.neighbors + 1), 2) & ...
        any(above(Gpc.faces.neighbors + 1), 2));
end
plot_contour_ext_counts = cellfun(@numel, plot_contour_ext);
plot_contour_ext_pos = cumsum([1; plot_contour_ext_counts]);
plot_contour_ext_faces = vertcat(plot_contour_ext{:});
plot_contour_int_counts = cellfun(@numel, plot_contour_int);
plot_contour_int_pos = cumsum([1; plot_contour_int_counts]);
plot_contour_int_faces = vertcat(plot_contour_int{:});

hist_values = [-2; -0.6; -0.4; 0; 0.4; 0.6; 2];
hist_limits = [-1, 1];
hist_centers = linspace(hist_limits(1), hist_limits(2), 3);
[hist_counts, hist_centers] = hist(hist_values, hist_centers);
hist_log_values = logspace(0, 3, 30)';
hist_log_limits = [1, 1000];
hist_log_centers = linspace(log10(hist_log_limits(1)), ...
                            log10(hist_log_limits(2)), 7);
[hist_log_counts, hist_log_centers] = hist( ...
    log10(hist_log_values), hist_log_centers);

Gvol = computeGeometry(cartGrid([2, 2, 2], [2, 2, 2]));
volume_cell_data = Gvol.cells.centroids(:, 1) + ...
                   Gvol.cells.centroids(:, 2) + ...
                   Gvol.cells.centroids(:, 3);
volume_boundary_faces = boundaryFaces(Gvol);
volume_boundary_owners = sum( ...
    Gvol.faces.neighbors(volume_boundary_faces, :), 2);
volume_source_coordinates = [Gvol.cells.centroids; ...
    Gvol.faces.centroids(volume_boundary_faces, :)];
volume_source_values = [volume_cell_data; ...
    volume_cell_data(volume_boundary_owners)];
volume_upper = max(volume_source_coordinates);
volume_upper = min([volume_upper; max(Gvol.nodes.coords)]);
volume_lower = min(volume_source_coordinates);
volume_lower = max([volume_lower; min(Gvol.nodes.coords)]);
volume_mesh = Gvol.cartDims + 2;
volume_axis_x = linspace(volume_lower(1), volume_upper(1), ...
                         volume_mesh(1) + 1);
volume_axis_y = linspace(volume_lower(2), volume_upper(2), ...
                         volume_mesh(2) + 1);
volume_axis_z = linspace(volume_lower(3), volume_upper(3), ...
                         volume_mesh(3) + 1);
[volume_x, volume_y, volume_z] = meshgrid( ...
    volume_axis_x, volume_axis_y, volume_axis_z);
volume_interpolator = scatteredInterpolant( ...
    volume_source_coordinates(:, 1), ...
    volume_source_coordinates(:, 2), ...
    volume_source_coordinates(:, 3), volume_source_values);
volume_samples_ref = volume_interpolator(volume_x, volume_y, volume_z);
[volume_isovalues, volume_isovalues] = hist(volume_cell_data, 3); %#ok

save(fullfile(thisDir, 'ref_plotting_phase1.mat'), '-v7', ...
    'b2_faces', 'b2_cells', ...
    'g3_boundary', 'g3_node_ids', 'g3_face_pos', 'g3_face_nodes', ...
    'g3_vertices', 'g3_bottom', 'g3_outline_edges', ...
    'cg_boundary', 'cg_fine_faces', 'cg_groups', ...
    'cell_data_faces', 'cell_data_owners', 'cell_data_values', ...
    'node_data_ids', 'line_cells', 'line_values', ...
    'line_cell_data_xy', 'line_grid_xy', ...
    'well_cells', 'well_completions', 'well_head', 'well_heel', ...
    'well_trajectory', 'well_radius', ...
    'fault_counts', 'fault_faces', 'fault_groups', ...
    'fault_label_positions', 'outline_faces2', 'outline_faces3', ...
    'block_id', 'block_neighbors', 'block_boundary_faces', ...
    'neighbor_face_pos', 'neighbor_boundary_faces', ...
    'slice_cells1', 'slice_faces1', 'slice_values1', ...
    'slice_cells2', 'slice_faces2', 'slice_values2', ...
    'slice_cells3', 'slice_faces3', 'slice_values3', ...
    'interp_node_data', 'interp_query', 'interp_affine', ...
    'interp_cell_data', 'interp_cells_to_nodes', ...
    'surface_cell_data', 'surface_node_values', 'surface_cells', ...
    'surface_node_ids', 'surface_cell_nodes', 'surface_cell_pos', ...
    'surface_vertices', 'contour_cell_data', 'contour_dims', ...
    'contour_x', 'contour_y', 'contour_values', ...
    'plot_contour_data', 'plot_contour_count', 'plot_contour_levels', ...
    'plot_contour_ext_pos', 'plot_contour_ext_faces', ...
    'plot_contour_int_pos', 'plot_contour_int_faces', ...
    'hist_values', 'hist_limits', 'hist_counts', 'hist_centers', ...
    'hist_log_values', 'hist_log_limits', 'hist_log_counts', ...
    'hist_log_centers', 'volume_cell_data', ...
    'volume_boundary_faces', 'volume_boundary_owners', ...
    'volume_source_coordinates', 'volume_source_values', ...
    'volume_axis_x', 'volume_axis_y', 'volume_axis_z', ...
    'volume_x', 'volume_y', 'volume_z', 'volume_samples_ref', ...
    'volume_isovalues');
fprintf('Wrote plotting Phase-1 topology reference\n');


function [node_ids, face_pos, face_nodes, vertices] = compactFaceTopology(G, faces)
    counts = G.faces.nodePos(faces + 1) - G.faces.nodePos(faces);
    ix = mcolon(G.faces.nodePos(faces), G.faces.nodePos(faces + 1) - 1);
    global_nodes = G.faces.nodes(ix);
    node_ids = unique(global_nodes, 'sorted');
    local = zeros(G.nodes.num, 1);
    local(node_ids) = 1:numel(node_ids);
    face_nodes = local(global_nodes);
    face_pos = cumsum([1; counts]);
    vertices = G.nodes.coords(node_ids, :);
end


function edges = collectionOutline(G, faces)
    edges = zeros(0, 2);
    for i = 1:numel(faces)
        f = faces(i);
        nodes = G.faces.nodes(G.faces.nodePos(f):G.faces.nodePos(f + 1)-1);
        edges = [edges; nodes, nodes([2:end, 1])]; %#ok<AGROW>
    end
    sorted = sort(edges, 2);
    [unique_edges, ~, index] = unique(sorted, 'rows');
    counts = accumarray(index, 1);
    edges = unique_edges(counts == 1, :);
    edges = sortrows(edges);
end


function [cells, faces, values] = sliceReference(G, data, slice_index, dim)
    [IX, IY, IZ] = ndgrid(1:G.cartDims(1), ...
                          1:G.cartDims(2), ...
                          1:G.cartDims(3));
    if dim == 1
        cart = sub2ind(G.cartDims, IX(slice_index,:,:), ...
                      IY(slice_index,:,:), IZ(slice_index,:,:));
        face_tag = 1;
    elseif dim == 2
        cart = sub2ind(G.cartDims, IX(:,slice_index,:), ...
                      IY(:,slice_index,:), IZ(:,slice_index,:));
        face_tag = 3;
    else
        cart = sub2ind(G.cartDims, IX(:,:,slice_index), ...
                      IY(:,:,slice_index), IZ(:,:,slice_index));
        face_tag = 5;
    end
    cells = cart2active(G, cart(:));
    cellno = rldecode(1:G.cells.num, diff(G.cells.facePos), 2)';
    selected_cells = false(G.cells.num, 1);
    selected_cells(cells) = true;
    mask = selected_cells(cellno) & G.cells.faces(:, 2) == face_tag;
    faces = G.cells.faces(mask, 1);
    owners = cellno(mask);
    values = data(owners);
end
