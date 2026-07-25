%% Dump Phase 365 reference: processNNC (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p365_nnc.mat.
%
% Pins the PyMRST `process_nnc` port against MRST's `processNNC` on a 4x4x3
% Cartesian grid with a deterministic NNC list that mixes genuine non-neighbour
% connections with pairs that are already faces (to exercise excludeExisting).
% The comparison confirms the logical->linear->active mapping (column-major
% sub2ind) and the existing-face exclusion match MRST exactly.

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([4, 4, 3], [4, 4, 3]));

% [i1 j1 k1  i2 j2 k2  trans]
NNC = [ 1 1 1   4 4 3   0.10; ...   % opposite corners, genuine NNC
        1 1 1   2 1 1   0.20; ...   % already a face (x-neighbour)
        2 2 2   2 2 3   0.30; ...   % already a face (z-neighbour)
        1 2 1   3 4 2   0.40; ...   % genuine NNC
        4 4 1   1 1 3   0.50 ];     % genuine NNC

nnc_all  = processNNC(G, NNC);                 % keep all (both active)
nnc_excl = processNNC(G, NNC, true);           % drop existing faces

cells_all  = nnc_all.cells;    trans_all  = nnc_all.trans;
cells_excl = nnc_excl.cells;   trans_excl = nnc_excl.trans;

save(fullfile(thisDir, 'ref_p365_nnc.mat'), '-v7', ...
     'cells_all', 'trans_all', 'cells_excl', 'trans_excl');

fprintf('Wrote ref_p365_nnc.mat to %s\n', thisDir);
fprintf('  all: %d connections, excludeExisting: %d\n', ...
        size(cells_all, 1), size(cells_excl, 1));
