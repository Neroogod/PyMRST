%% Reference data for large faulted corner-point decks

thisDir = fileparts(mfilename('fullpath'));

%% Tier 1: full sorted volumes at a size that stays small on disk.
dims_full = [40, 40, 10];
G = computeGeometry(processGRDECL(oneSlopingFault(dims_full, 5)));
full_cells = G.cells.num;
full_faces = G.faces.num;
full_nodes = G.nodes.num;
full_volume = sum(G.cells.volumes);
full_volumes_sorted = sort(G.cells.volumes);
full_face_areas_sorted = sort(G.faces.areas);
full_boundary_faces = nnz(any(G.faces.neighbors == 0, 2));

%% Tier 2: aggregates only, at a size worth exercising but not storing.
dims_big = [60, 60, 20];
H = computeGeometry(processGRDECL(oneSlopingFault(dims_big, 5)));
big_cells = H.cells.num;
big_faces = H.faces.num;
big_nodes = H.nodes.num;
big_volume = sum(H.cells.volumes);
big_min_volume = min(H.cells.volumes);
big_max_volume = max(H.cells.volumes);
big_boundary_faces = nnz(any(H.faces.neighbors == 0, 2));
% 21 samples of the sorted values fingerprint the distribution without storing
% every cell. Indices are integer arithmetic so Python reproduces them exactly.
sample = @(v) reshape(v(1 + floor((0:20) * (numel(v) - 1) / 20)), 1, []);
big_volume_quantiles = sample(sort(H.cells.volumes));
big_area_quantiles = sample(sort(H.faces.areas));

%% Disconnected lattice: every third i- and j-plane switched off.
dims_iso = [30, 30, 8];
[ii, jj, ~] = ndgrid(1:dims_iso(1), 1:dims_iso(2), 1:dims_iso(3));
act = (mod(ii - 1, 3) ~= 2) & (mod(jj - 1, 3) ~= 2);
deck = oneSlopingFault(dims_iso, 0);
deck.ACTNUM = int32(act(:));
parts = processGRDECL(deck);
iso_components = numel(parts);
iso_cells = sort(arrayfun(@(p) p.cells.num, parts));
iso_active = nnz(act);

save(fullfile(thisDir, 'ref_grid_large_deck.mat'), '-v7', ...
     'dims_full', 'full_cells', 'full_faces', 'full_nodes', 'full_volume', ...
     'full_volumes_sorted', 'full_face_areas_sorted', 'full_boundary_faces', ...
     'dims_big', 'big_cells', 'big_faces', 'big_nodes', 'big_volume', ...
     'big_min_volume', 'big_max_volume', 'big_boundary_faces', ...
     'big_volume_quantiles', 'big_area_quantiles', ...
     'dims_iso', 'iso_components', 'iso_cells', 'iso_active');
fprintf('Wrote large-deck reference to %s\n', thisDir);
