% Reference for validate_compute_vertical_grid_intersection.py
out = struct();
G = computeGeometry(cartGrid([4,4,3],[4,4,3]));
G = addBoundingBoxFields(G, 'faces', true);

segs = { [0.5 0.5; 3.5 0.5], ...
         [0.2 0.2; 3.8 3.8], ...
         [0.5 0.5; 2.5 0.5; 2.5 3.5], ...
         [-1 0.5; 5 0.5] };
names = {'straight','diagonal','bend','spanning'};

for k = 1:numel(segs)
   p = computeVerticalGridIntersection(G, segs{k});
   r = struct();
   r.cellIx    = p.cellIx(:);
   r.cumlength = p.cumlength(:);
   r.nPoints   = size(p.coords3D, 1);
   r.nPoly     = size(p.nodes, 1);
   r.maxWidth  = size(p.nodes, 2);
   % Order-independent fingerprints of the geometry.
   r.c3sorted  = sortrows(round(p.coords3D, 9));
   r.c2sorted  = sortrows(round(p.coords2D, 9));
   out.(names{k}) = r;
end

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_vertical_intersection.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_vertical_intersection.mat\n');
