% Reference output of MRST's createGridMappings, for validate_create_grid_mappings.py
out = struct();
names = {'c322','c33'};
cases = {{[3,2,2],[3,2,2]}, {[3,3],[3,3]}};
for k = 1:numel(cases)
   G = computeGeometry(cartGrid(cases{k}{1}, cases{k}{2}));
   r = struct();
   r.w = createGridMappings(G);
   r.cellFaces = double(G.cells.faces(:,1));
   r.facePos = double(G.cells.facePos(:));
   r.numFaces = G.faces.num;
   r.numNodes = G.nodes.num;
   r.faceNodes = double(G.faces.nodes(:));
   r.nodePos = double(G.faces.nodePos(:));
   r.coords = G.nodes.coords;
   out.(names{k}) = r;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_grid_mappings.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_grid_mappings.mat\n');
