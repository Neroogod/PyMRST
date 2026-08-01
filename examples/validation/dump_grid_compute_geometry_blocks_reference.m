%% Reference data for computeGeometry's MaxBlockSize option

thisDir = fileparts(mfilename('fullpath'));

%% MRST's private partition_cells arithmetic, reproduced inline so the block
% boundaries can be compared against the Python port.
cases = [10 1; 10 2; 10 3; 10 4; 10 7; 10 10; 10 1000; 7 3; 100 7; 997 64; 1 1];
starts = cell(size(cases, 1), 1);
stops = cell(size(cases, 1), 1);
for k = 1:size(cases, 1)
    M = cases(k, 1);
    B = cases(k, 2);
    if isempty(B) || (B < 0)
        B = M;
    end
    B = min(B, floor(M / ceil(M / B)));
    L = floor(M / B);
    R = mod(M, B);
    p = cumsum([1, repmat(B, [1, L]) + [ones([1, R]), zeros([1, L - R])]]);
    blocks = [p(1:(end - 1)); p(2:end) - 1];
    starts{k} = blocks(1, :) - 1;                  % zero-based, inclusive
    stops{k} = blocks(2, :);                       % zero-based, exclusive
end
block_starts = starts;
block_stops = stops;
block_cases = cases;

%% MaxBlockSize must not change the numbers computeGeometry produces.
G = cartGrid([5, 4, 3], [5, 4, 3]);
rng(7);
pts = G.nodes.coords;
lo = min(pts, [], 1);
hi = max(pts, [], 1);
inner = all(pts > lo + 1e-9 & pts < hi - 1e-9, 2);
pts(inner, :) = pts(inner, :) + 0.15 * (rand(nnz(inner), 3) - 0.5);
G.nodes.coords = pts;
node_coords = pts;

S = computeGeometry(G, 'MaxBlockSize', []);
single_volumes = S.cells.volumes;
single_centroids = S.cells.centroids;
single_face_areas = S.faces.areas;

blocked_max_diff = 0;
for B = [1, 2, 3, 7, 13, 59, 60, 61]
    Bk = computeGeometry(G, 'MaxBlockSize', B);
    blocked_max_diff = max([blocked_max_diff; ...
        abs(Bk.cells.volumes - single_volumes); ...
        abs(Bk.cells.centroids(:) - single_centroids(:)); ...
        abs(Bk.faces.areas - single_face_areas)]);
end

save(fullfile(thisDir, 'ref_grid_compute_geometry_blocks.mat'), '-v7', ...
     'block_cases', 'block_starts', 'block_stops', 'node_coords', ...
     'single_volumes', 'single_centroids', 'single_face_areas', ...
     'blocked_max_diff');
fprintf('Wrote computeGeometry MaxBlockSize reference to %s\n', thisDir);
