% Reference for validate_simple_equilibrium.py
out = struct();
G = computeGeometry(cartGrid([1,1,10],[1,1,10]));
out.two    = simpleEquilibrium(G, 5);
out.three  = simpleEquilibrium(G, [3, 7]);
out.strad  = simpleEquilibrium(G, 3.25);
out.above  = simpleEquilibrium(G, -100);
out.vec    = simpleEquilibrium(G, 5, [0 0 1]);
Gx = computeGeometry(cartGrid([10,1,1],[10,1,1]));
out.axis1  = simpleEquilibrium(Gx, 5, 1);
% 2-D is deliberately absent: simpleEquilibrium's default vector is [0 0 1],
% which cannot multiply a 2-D grid's (n,2) centroids. Recorded here rather
% than worked around, and asserted by the validator.
G2 = computeGeometry(cartGrid([1,10],[1,10]));
try
   simpleEquilibrium(G2, 5);
   out.twod_ran = 1;
catch err
   out.twod_ran = 0;
   out.twod_msg = err.message;
end
% A distorted grid: face-centroid extents stop matching node extents.
Gt = cartGrid([3,3,4],[3,3,4]);
c = Gt.nodes.coords;
rng(2);
interior = all(c > 1e-9 & c < repmat([3,3,4], size(c,1), 1) - 1e-9, 2);
c(interior,:) = c(interior,:) + (rand(nnz(interior),3)*0.3 - 0.15);
Gt.nodes.coords = c; Gt = computeGeometry(Gt);
out.twist  = simpleEquilibrium(Gt, [1.5, 2.5]);
out.twistcoords = Gt.nodes.coords;
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_equil.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_equil.mat\n');
