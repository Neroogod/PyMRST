% Reference for validate_transform_3d_to_2d_grid.py
%
% transform3Dto2Dgrid reads faces.numNodes and cells.numFaces, which modern
% MRST grids no longer carry, so it cannot run on the output of
% computeGeometry(cartGrid(...)) as shipped. The two legacy fields are
% supplied here to revive it; see the port's docstring.
out = struct();
cases = {{[1,4,3],[1,4,3]}, {[4,1,3],[4,1,3]}, {[4,3,1],[4,3,1]}, {[1,5,4],[3,5,4]}};
names = {'x','y','z','thick'};
for k = 1:numel(cases)
   G = computeGeometry(cartGrid(cases{k}{1}, cases{k}{2}));
   G.faces.numNodes = diff(G.faces.nodePos);
   G.cells.numFaces = diff(G.cells.facePos);
   G2 = transform3Dto2Dgrid(G);
   r = struct();
   r.numCells = G2.cells.num;
   r.numFaces = G2.faces.num;
   r.numNodes = G2.nodes.num;
   r.cartDims = G2.cartDims;
   r.volumes  = G2.cells.volumes;
   r.fcent    = G2.faces.centroids;
   r.fareas   = G2.faces.areas;
   r.fneigh   = G2.faces.neighbors;
   r.ccent    = G2.cells.centroids;
   r.ncoords  = G2.nodes.coords;
   out.(names{k}) = r;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_transform_2d.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_transform_2d.mat\n');
