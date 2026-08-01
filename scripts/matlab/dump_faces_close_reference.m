% Reference for validate_get_faces_close_to_segment.py
out = struct();
G = computeGeometry(cartGrid([6,6,4],[6,6,4]));
G = addBoundingBoxFields(G, 'faces', true);

segs = { [0.3 0.4 0.5; 5.7 5.6 3.5], ...
         [0.5 0.5 0.5; 5.5 0.5 0.5], ...
         [2.5 0.2 1.5; 2.5 5.8 1.5], ...
         [0.5 3.0 2.0; 3.0 4.2 2.0; 5.5 3.0 2.0] };
names = {'diag','axis_x','axis_y','poly'};
for k = 1:numel(segs)
   ix = getFacesCloseToSegment(G, segs{k});
   out.(names{k}) = G.faces.centroids(ix, :);
end

out.fac_half = G.faces.centroids(getFacesCloseToSegment(G, segs{2}, 'fac', 0.5), :);
out.fac_two  = G.faces.centroids(getFacesCloseToSegment(G, segs{2}, 'fac', 2.0), :);
out.projz    = G.faces.centroids(getFacesCloseToSegment(G, [1.5 1.5 0.2; 4.5 4.5 0.4], 'projVec', [0 0 1]), :);
out.far      = G.faces.centroids(getFacesCloseToSegment(G, [100 100 100; 200 200 200]), :);

G2 = computeGeometry(cartGrid([6,6],[6,6]));
G2 = addBoundingBoxFields(G2, 'faces', true);
out.twod = G2.faces.centroids(getFacesCloseToSegment(G2, [0.5 0.5; 5.5 5.5]), :);

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_faces_close.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_faces_close.mat\n');
