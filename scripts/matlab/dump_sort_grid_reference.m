% Reference output of MRST's sortGrid, for validate_sort_grid.py.
mrstModule add libgeometry
out = struct();
cases = {{[3,3],[3,3]}, {[4,3],[4,3]}, {[2,2,2],[2,2,2]}, {[3,2,2],[3,2,2]}};
names = {'c33','c43','c222','c322'};
for k = 1:numel(cases)
   G = cartGrid(cases{k}{1}, cases{k}{2});
   S = sortGrid(G);
   r = struct();
   r.coords    = S.nodes.coords;
   r.faceNodes = double(S.faces.nodes(:));
   r.nodePos   = double(S.faces.nodePos(:));
   r.neighbors = double(S.faces.neighbors);
   r.cellFaces = double(S.cells.faces(:,1));
   r.facePos   = double(S.cells.facePos(:));
   r.griddim   = S.griddim;
   out.(names{k}) = r;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_sort_grid.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_sort_grid.mat\n');
