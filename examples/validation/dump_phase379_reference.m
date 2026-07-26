%% Dump Phase 379 reference: removeShortEdges (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p379_shortedges.mat.
%
% Pins the PyMRST `remove_short_edges` port against MRST's `removeShortEdges` on
% a pentagon whose two top nodes are 1e-4 apart (a sliver edge). Collapsing edges
% below 1e-3 turns it into a quad. The resulting cell/face/node counts and sorted
% node coordinates are compared (the sliver's endpoints merge to their average).

thisDir = fileparts(mfilename('fullpath'));

p = [0 0; 1 0; 1 1; 0.5 1.0001; 0.5 1.0];
t = [1 2 3 4 5];
G = tessellationGrid(p, t);
G = removeShortEdges(G, 1e-3);

cells_num = G.cells.num;
faces_num = G.faces.num;
nodes_num = G.nodes.num;
coords    = sortrows(round(G.nodes.coords, 8));

save(fullfile(thisDir, 'ref_p379_shortedges.mat'), '-v7', ...
     'cells_num', 'faces_num', 'nodes_num', 'coords');

fprintf('Wrote ref_p379_shortedges.mat to %s\n', thisDir);
fprintf('  pentagon -> %d cells, %d faces, %d nodes\n', ...
        cells_num, faces_num, nodes_num);
