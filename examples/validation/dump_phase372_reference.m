%% Dump Phase 372 reference: triangulateFaces (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p372_trifaces.mat.
%
% Pins the PyMRST `triangulate_faces` port against MRST's `triangulateFaces` on a
% 2x2x1 Cartesian grid (all quad faces -> triangles). Face renumbering may
% differ, so the check uses the face/node counts, the sorted multiset of
% neighbour pairs (outside as -1), and the sorted set of added centroid-node
% coordinates - which pin both the topology and the geometry of the split.

thisDir = fileparts(mfilename('fullpath'));

G  = computeGeometry(cartGrid([2, 2, 1], [2, 2, 1]));
nn0 = G.nodes.num;
G2 = triangulateFaces(G);

faces_num = G2.faces.num;
nodes_num = G2.nodes.num;

nb = double(G2.faces.neighbors);
nb(nb == 0) = -1;
nb = sort(nb, 2);
pairs = sortrows(nb);

% the added centroid nodes, sorted for an order-independent comparison
new_nodes = sortrows(G2.nodes.coords(nn0+1:end, :));

save(fullfile(thisDir, 'ref_p372_trifaces.mat'), '-v7', ...
     'faces_num', 'nodes_num', 'pairs', 'new_nodes');

fprintf('Wrote ref_p372_trifaces.mat to %s\n', thisDir);
fprintf('  faces %d, nodes %d, new centroid nodes %d\n', ...
        faces_num, nodes_num, size(new_nodes, 1));
