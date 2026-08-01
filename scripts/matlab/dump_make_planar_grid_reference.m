% Reference output of MRST's makePlanarGrid, for validate_make_planar_grid.py
out = struct();
H = computeGeometry(cartGrid([3,3,2],[3,3,2]));

% The faces on the top of the parent.
isflat = abs(H.faces.normals(:,3)) > sum(abs(H.faces.normals(:,1:2)), 2);
top = find(isflat & abs(H.faces.centroids(:,3) - max(H.faces.centroids(:,3))) < 1e-9);

G = makePlanarGrid(H, top);
out.top.faceNodes  = double(G.faces.nodes(:));
out.top.nodePos    = double(G.faces.nodePos(:));
out.top.neighbors  = double(G.faces.neighbors);
out.top.cellFaces  = double(G.cells.faces(:,1));
out.top.facePos    = double(G.cells.facePos(:));
out.top.numFaces   = G.faces.num;
out.top.numCells   = G.cells.num;
out.top.numNodes   = G.nodes.num;
out.top.sel        = top;

% Parent face node lists, so the comparison can be made numbering-independent.
out.parent.faceNodes = double(H.faces.nodes(:));
out.parent.nodePos   = double(H.faces.nodePos(:));
out.parent.coords    = H.nodes.coords;

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_make_planar_grid.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_make_planar_grid.mat (%d cells, %d faces)\n', G.cells.num, G.faces.num);
