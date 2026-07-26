%% Dump Phase 373 reference: findMatchingFaces (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p373_matchfaces.mat.
%
% Pins the PyMRST `find_matching_faces` port against MRST's `findMatchingFaces`.
% A 3x3x2 grid is cut along a deterministic set of internal faces (making
% coincident boundary-face pairs), then the matching pairs are recovered. Face
% renumbering differs, so the check uses the pair count and the sorted set of
% matched-pair midpoint coordinates (order- and renumbering-independent geometry).

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3, 2], [3, 3, 2]));
internal = find(all(G.faces.neighbors ~= 0, 2));
cut = internal(1:2:end);
cut0 = cut - 1;

G2 = makeInternalBoundary(G, cut);
G2 = computeGeometry(G2);

N = findMatchingFaces(G2);
num_pairs = size(N, 1);

% midpoint of each matched pair's (shared) face, sorted for comparison
mids = (G2.faces.centroids(N(:,1), :) + G2.faces.centroids(N(:,2), :)) / 2;
mids = sortrows(round(mids, 10));

save(fullfile(thisDir, 'ref_p373_matchfaces.mat'), '-v7', ...
     'num_pairs', 'mids', 'cut0');

fprintf('Wrote ref_p373_matchfaces.mat to %s\n', thisDir);
fprintf('  cut %d internal faces -> %d matched pairs\n', numel(cut), num_pairs);
