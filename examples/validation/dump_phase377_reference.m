%% Dump Phase 377 reference: glue2DGrid (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p377_glue.mat.
%
% Pins the PyMRST `glue_2d_grid` port against MRST's `glue2DGrid` on two cases:
%   (A) two matching unit squares sharing an edge (node-conforming), and
%   (B) a coarse square glued to two stacked squares (needs edge splitting to
%       conform).
% Face renumbering differs, so the check uses the cell/face/node counts and the
% sorted neighbour-pair multiset (outside as -1) - which pin the glued topology.

thisDir = fileparts(mfilename('fullpath'));

topo = @(G) deal(G.cells.num, G.faces.num, G.nodes.num, ...
                 sortrows(sort(double(G.faces.neighbors) + ...
                          (double(G.faces.neighbors) == 0) * -1, 2)));

% -- case A: matching squares -------------------------------------------------
gA1 = tessellationGrid([0 0; 1 0; 1 1; 0 1], [1 2 3 4]);
gA2 = tessellationGrid([1 0; 2 0; 2 1; 1 1], [1 2 3 4]);
GA  = glue2DGrid(gA1, gA2);
[A_c, A_f, A_n, A_pairs] = topo(GA);

% -- case B: coarse vs two stacked (non-conforming) ---------------------------
gB1 = tessellationGrid([0 0; 1 0; 1 1; 0 1], [1 2 3 4]);
gB2 = tessellationGrid([1 0; 2 0; 2 .5; 1 .5; 2 1; 1 1], [1 2 3 4; 4 3 5 6]);
GB  = glue2DGrid(gB1, gB2);
[B_c, B_f, B_n, B_pairs] = topo(GB);

save(fullfile(thisDir, 'ref_p377_glue.mat'), '-v7', ...
     'A_c', 'A_f', 'A_n', 'A_pairs', 'B_c', 'B_f', 'B_n', 'B_pairs');

fprintf('Wrote ref_p377_glue.mat to %s\n', thisDir);
fprintf('  case A: %d cells, %d faces, %d nodes\n', A_c, A_f, A_n);
fprintf('  case B: %d cells, %d faces, %d nodes\n', B_c, B_f, B_n);
