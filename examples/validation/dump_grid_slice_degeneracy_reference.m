%% Reference data for sliceGrid degeneracies: in-plane and collapsed faces

thisDir = fileparts(mfilename('fullpath'));

%% Pinched cell whose collapsed face lies in the cut plane.
% The top face is squeezed onto x=0.5, so the plane x=0.5 contains that whole
% face while still passing through the cell interior.
G = cartGrid([1, 1, 1], [1, 1, 1]);
G.nodes.coords(G.nodes.coords(:, 3) > 0.5, 1) = 0.5;
G = computeGeometry(G);
pinch_coords = G.nodes.coords;
pinch_volume_in = sum(G.cells.volumes);

[P, pix, P_slice] = sliceGrid(G, [0.5, 0.5, 0.5], 'normal', [1, 0, 0]);
pinch_cells_num = P.cells.num;
pinch_total_volume = sum(P.cells.volumes);
pinch_volumes_sorted = sort(P.cells.volumes);
pinch_face_status = pix.new.faces;
pinch_cell_status = pix.new.cells;
pinch_face_areas = sort(P.faces.areas);
pinch_degenerate_faces = nnz(P.faces.areas <= 1e-12);
pinch_face_areas_nz = sort(P.faces.areas(P.faces.areas > 1e-12));
pinch_slice_num_cells = 0;
if ~isempty(P_slice)
    pinch_slice_num_cells = P_slice.cells.num;
end

%% Cut snapped onto an existing column of faces: nothing is split, but the
% coinciding faces are still the slice.
S = computeGeometry(cartGrid([2, 1, 1], [2, 1, 1]));
snap_volume_in = sum(S.cells.volumes);
[Q, qix, Q_slice] = sliceGrid(S, [1, -1, 0; 1, 2, 0], 'cutDir', [0, 0, 1]);
snap_cells_num = Q.cells.num;
snap_total_volume = sum(Q.cells.volumes);
snap_face_status = qix.new.faces;
snap_cell_status = qix.new.cells;
snap_slice_num_cells = 0;
snap_slice_coords = [];
if ~isempty(Q_slice)
    snap_slice_num_cells = Q_slice.cells.num;
    snap_slice_coords = sortrows(Q_slice.nodes.coords);
end

save(fullfile(thisDir, 'ref_grid_slice_degeneracy.mat'), '-v7', ...
     'pinch_coords', 'pinch_volume_in', 'pinch_cells_num', ...
     'pinch_total_volume', 'pinch_volumes_sorted', 'pinch_face_status', ...
     'pinch_cell_status', 'pinch_slice_num_cells', ...
     'pinch_face_areas', 'pinch_degenerate_faces', 'pinch_face_areas_nz', ...
     'snap_volume_in', 'snap_cells_num', 'snap_total_volume', ...
     'snap_face_status', 'snap_cell_status', 'snap_slice_num_cells', ...
     'snap_slice_coords');
fprintf('Wrote sliceGrid degeneracy reference to %s\n', thisDir);
