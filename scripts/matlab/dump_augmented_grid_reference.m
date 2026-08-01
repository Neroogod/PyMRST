% Reference for validate_create_augmented_grid.py
out = struct();
cases = {{[3,2,2],[3,2,2]}, {[3,3],[3,3]}, {[2,2,2],[2,2,2]}};
names = {'c322','c33','c222'};
for k = 1:numel(cases)
   G = computeGeometry(cartGrid(cases{k}{1}, cases{k}{2}));
   A = createAugmentedGrid(G);
   r = struct();
   r.cellNodes  = double(A.cells.nodes(:));
   r.cellNodePos= double(A.cells.nodePos(:));
   r.nodeFaces  = double(A.nodes.faces(:));
   r.nodeFacePos= double(A.nodes.facePos(:));
   r.nodeCells  = double(A.nodes.cells(:));
   r.nodeCellPos= double(A.nodes.cellPos(:));
   r.coords     = A.nodes.coords;
   if isfield(A, 'edges')
      r.numEdges = A.edges.num;
      r.edgeNodes= double(A.edges.nodes(:));
      r.faceEdges= double(A.faces.edges(:));
      r.faceEdgePos = double(A.faces.edgePos(:));
      r.edgeSign = double(A.faces.edgeSign(:));
   else
      r.numEdges = 0;
   end
   out.(names{k}) = r;
end

% Does an unreferenced node break MRST's position arrays? Claim under test:
% map2Pos uses find(diff(keys)==1) and assumes no key is skipped.
G = computeGeometry(cartGrid([2,2,2],[2,2,2]));
G.nodes.coords(end+1, :) = [99 99 99];
G.nodes.num = G.nodes.num + 1;
try
   A = createAugmentedGrid(G);
   out.orphan.ran = 1;
   out.orphan.numNodes = A.nodes.num;
   out.orphan.lenCellPos = numel(A.nodes.cellPos);
   out.orphan.lenFacePos = numel(A.nodes.facePos);
   out.orphan.expected = A.nodes.num + 1;
catch err
   out.orphan.ran = 0;
   out.orphan.msg = err.message;
end

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_augmented_grid.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_augmented_grid.mat\n');
if out.orphan.ran
  fprintf('ORPHAN: nodes=%d cellPos=%d facePos=%d expected=%d\n', ...
     out.orphan.numNodes, out.orphan.lenCellPos, out.orphan.lenFacePos, out.orphan.expected);
else
  fprintf('ORPHAN: errored -> %s\n', out.orphan.msg);
end
