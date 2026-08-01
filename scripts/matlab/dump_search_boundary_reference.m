% Reference for validate_search_for_boundary_faces.py
out = struct();
G = computeGeometry(cartGrid([4,3,2],[4,3,2]));
G2 = computeGeometry(cartGrid([4,3],[4,3]));

sides = {'west','east','south','north','top','bottom'};
for k = 1:numel(sides)
   ix = searchForBoundaryFaces(G, sides{k});
   out.(['s_' sides{k}]) = G.faces.centroids(ix, :);
end

% Restricted cases; MATLAB indices are 1-based.
out.r_k1     = G.faces.centroids(searchForBoundaryFaces(G, 'west', [], [], 1), :);
out.r_i1     = G.faces.centroids(searchForBoundaryFaces(G, 'west', 1, [], []), :);
out.r_i2     = G.faces.centroids(searchForBoundaryFaces(G, 'west', 2, [], []), :);
out.r_combo  = G.faces.centroids(searchForBoundaryFaces(G, 'top', [1 2], 3, []), :);
out.t2d_west = G2.faces.centroids(searchForBoundaryFaces(G2, 'west'), :);
out.t2d_north= G2.faces.centroids(searchForBoundaryFaces(G2, 'north'), :);

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_search_boundary.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_search_boundary.mat\n');
