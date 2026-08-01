%% Advanced deterministic plotting fixtures for PyMRST.
thisDir = fileparts(mfilename('fullpath'));
workspace = fileparts(fileparts(fileparts(thisDir)));
mrstRoot = fullfile(workspace, 'mrst-2024a-Geomech');
addpath(mrstRoot);
run(fullfile(mrstRoot, 'startup.m'));

% Faulted corner-point boundary patch topology.
Gfault = computeGeometry(processGRDECL(oneSlopingFault([4, 2, 2], 2)));
fault_boundary = boundaryFaces(Gfault);
[fault_face_pos, fault_face_nodes, fault_node_coords] = ...
    patchTopology(Gfault, fault_boundary);

% General triangular-prism polyhedral grid.
points = [0, 0; 2, 0; 2, 2; 0, 2; 1, 1];
triangles = [1, 2, 5; 2, 3, 5; 3, 4, 5; 4, 1, 5];
Gpoly = makeLayeredGrid(triangleGrid(points, triangles), 2);
Gpoly = computeGeometry(Gpoly);
poly_boundary = boundaryFaces(Gpoly);
[poly_face_pos, poly_face_nodes, poly_node_coords] = ...
    patchTopology(Gpoly, poly_boundary);

% Slice through a Cartesian grid with one inactive cell.
Ginactive = computeGeometry(cartGrid([3, 2, 2], [3, 2, 2]));
Ginactive = removeCells(Ginactive, [2; 8]);
inactive_data = (1:Ginactive.cells.num)';
[inactive_slice_cells1, inactive_slice_faces1, inactive_slice_values1] = ...
    sliceReference(Ginactive, inactive_data, 2, 1);
[inactive_slice_cells2, inactive_slice_faces2, inactive_slice_values2] = ...
    sliceReference(Ginactive, inactive_data, 1, 2);
[inactive_slice_cells3, inactive_slice_faces3, inactive_slice_values3] = ...
    sliceReference(Ginactive, inactive_data, 2, 3);

% Isosurface statistics. Connectivity is renderer-dependent, so store
% invariants that permit comparison across MATLAB and VTK triangulations.
axis_values = linspace(0, 2, 7);
[iso_x, iso_y, iso_z] = meshgrid(axis_values, axis_values, axis_values);
iso_values = iso_x + iso_y + iso_z;
iso_level = 3;
iso_patch = isosurface(iso_x, iso_y, iso_z, iso_values, iso_level);
iso_vertex_count = size(iso_patch.vertices, 1);
iso_face_count = size(iso_patch.faces, 1);
iso_bounds = [min(iso_patch.vertices); max(iso_patch.vertices)];
iso_area = triangleArea(iso_patch.vertices, iso_patch.faces);

save(fullfile(thisDir, 'ref_plotting_advanced.mat'), '-v7', ...
    'fault_boundary', 'fault_face_pos', 'fault_face_nodes', ...
    'fault_node_coords', 'poly_boundary', 'poly_face_pos', ...
    'poly_face_nodes', 'poly_node_coords', 'inactive_slice_cells1', ...
    'inactive_slice_faces1', 'inactive_slice_values1', ...
    'inactive_slice_cells2', 'inactive_slice_faces2', ...
    'inactive_slice_values2', 'inactive_slice_cells3', ...
    'inactive_slice_faces3', 'inactive_slice_values3', 'axis_values', ...
    'iso_level', 'iso_vertex_count', 'iso_face_count', 'iso_bounds', ...
    'iso_area');
fprintf('Wrote advanced plotting reference\n');


function [face_pos, face_nodes, node_coords] = patchTopology(G, faces)
    counts = G.faces.nodePos(faces + 1) - G.faces.nodePos(faces);
    ix = mcolon(G.faces.nodePos(faces), G.faces.nodePos(faces + 1) - 1);
    face_nodes = G.faces.nodes(ix);
    face_pos = cumsum([1; counts]);
    node_coords = G.nodes.coords;
end


function [cells, faces, values] = sliceReference(G, data, slice_index, dim)
    [IX, IY, IZ] = ndgrid(1:G.cartDims(1), 1:G.cartDims(2), 1:G.cartDims(3));
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
    cells = cells(cells > 0);
    cellno = rldecode(1:G.cells.num, diff(G.cells.facePos), 2)';
    selected = false(G.cells.num, 1);
    selected(cells) = true;
    mask = selected(cellno) & G.cells.faces(:, 2) == face_tag;
    faces = G.cells.faces(mask, 1);
    values = data(cellno(mask));
end


function area = triangleArea(vertices, faces)
    a = vertices(faces(:, 2), :) - vertices(faces(:, 1), :);
    b = vertices(faces(:, 3), :) - vertices(faces(:, 1), :);
    area = .5*sum(sqrt(sum(cross(a, b, 2).^2, 2)));
end
