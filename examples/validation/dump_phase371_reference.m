%% Dump Phase 371 reference: tessellationGrid (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p371_tessgrid.mat.
%
% Pins the PyMRST `tessellation_grid` port against MRST's `tessellationGrid` on a
% deterministic 2x2 tessellation of quads over a 3x3 point grid. Face renumbering
% may differ, so the check uses the face/cell/node counts and the sorted
% multiset of neighbour pairs (outside as -1). p and the tessellation t are
% dumped so PyMRST builds from identical input.

thisDir = fileparts(mfilename('fullpath'));

% 3x3 point grid; node index = i + 3*j + 1 (1-based), i,j in 0..2
[I, J] = ndgrid(0:2, 0:2);
p = [I(:), J(:)];
idx = @(i, j) i + 3*j + 1;

% four quads, each [SW SE NE NW]
t = zeros(4, 4);
c = 0;
for J0 = 0:1
    for I0 = 0:1
        c = c + 1;
        t(c, :) = [idx(I0, J0), idx(I0+1, J0), idx(I0+1, J0+1), idx(I0, J0+1)];
    end
end

G = tessellationGrid(p, t);

faces_num = G.faces.num;
cells_num = G.cells.num;
nodes_num = G.nodes.num;

nb = double(G.faces.neighbors);
nb(nb == 0) = -1;
nb = sort(nb, 2);
pairs = sortrows(nb);

t0 = t - 1;         % 0-based for PyMRST

save(fullfile(thisDir, 'ref_p371_tessgrid.mat'), '-v7', ...
     'faces_num', 'cells_num', 'nodes_num', 'pairs', 'p', 't0');

fprintf('Wrote ref_p371_tessgrid.mat to %s\n', thisDir);
fprintf('  cells %d, faces %d, nodes %d\n', cells_num, faces_num, nodes_num);
