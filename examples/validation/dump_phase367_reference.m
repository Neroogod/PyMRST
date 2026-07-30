%% Dump Phase 367 reference: makeInternalBoundary (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p367_intbnd.mat.
%
% Pins the PyMRST `make_internal_boundary` port against MRST's
% `makeInternalBoundary` on a 3x3x2 grid with a deterministic set of internal
% faces cut. Face renumbering after remove/add differs between the two codes, so
% the comparison uses renumbering-independent quantities: the face and cell
% counts, the number of boundary faces, and the sorted multiset of
% neighbour pairs (outside encoded as -1). Those fully pin the topology.

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3, 2], [3, 3, 2]));

internal = find(all(G.faces.neighbors ~= 0, 2));
cut = internal(1:2:end);          % a deterministic subset (every other)

G2 = makeInternalBoundary(G, cut);

faces_num = G2.faces.num;
cells_num = G2.cells.num;

nb = G2.faces.neighbors;
nb(nb == 0) = -1;                 % outside -> -1 (match PyMRST convention)
nb = sort(nb, 2);                 % order-independent within a face
pairs = sortrows(nb);             % order-independent across faces
n_boundary = sum(any(G2.faces.neighbors == 0, 2));

cut0 = cut - 1;                   % 0-based, for PyMRST to select the same faces

save(fullfile(thisDir, 'ref_p367_intbnd.mat'), '-v7', ...
     'faces_num', 'cells_num', 'n_boundary', 'pairs', 'cut0');

% Tagged two-cell case pins replacement face tags and directional half-face
% tags copied from the original shared face.
Gt = cartGrid([2, 1, 1], [2, 1, 1]);
tag_cut = find(all(Gt.faces.neighbors ~= 0, 2), 1);
[Gt, tag_N] = makeInternalBoundary(Gt, tag_cut, 'tag', 77);
replacement_face_tags = Gt.faces.tag(tag_N(:));
replacement_half_tags = sortrows(Gt.cells.faces( ...
    ismember(Gt.cells.faces(:, 1), tag_N(:)), :));
save(fullfile(thisDir, 'ref_p367_intbnd.mat'), '-append', ...
     'replacement_face_tags', 'replacement_half_tags');

fprintf('Wrote ref_p367_intbnd.mat to %s\n', thisDir);
fprintf('  cut %d internal faces; faces %d -> %d, boundary faces = %d\n', ...
        numel(cut), G.faces.num, faces_num, n_boundary);
