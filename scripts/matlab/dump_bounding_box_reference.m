% Reference output of MRST's addBoundingBoxFields, for validate_add_bounding_box_fields.py
out = struct();
cases = {{[3,2,2],[3,2,2]}, {[3,3],[3,3]}, {[2,2,2],[4,6,10]}};
names = {'c322','c33','stretched'};
for k = 1:numel(cases)
   G = computeGeometry(cartGrid(cases{k}{1}, cases{k}{2}));
   G = addBoundingBoxFields(G, 'cells', true, 'faces', true);
   r = struct();
   r.cellBbox = G.cells.bbox;
   r.faceBbox = G.faces.bbox;
   out.(names{k}) = r;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_bounding_box.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_bounding_box.mat\n');
