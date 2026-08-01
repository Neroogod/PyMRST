% Reference for validate_tensor_map.py
out = struct();
cf = IndexArray(struct('cells', [1;1;2], 'faces', [4;8;8]));
c  = IndexArray(struct('cells', [1;2]));

m = TensorMap(); m.fromTbl = cf; m.toTbl = c; m.mergefds = {'cells'};
m = m.setup();
out.sum = struct('v', m.eval([1;10;100]), 'd1', m.dispind1(:), 'd2', m.dispind2(:));

m2 = TensorMap(); m2.fromTbl = c; m2.toTbl = cf; m2.mergefds = {'cells'};
m2 = m2.setup();
out.broadcast = struct('v', m2.eval([5;9]), 'd1', m2.dispind1(:), 'd2', m2.dispind2(:));

out.transpose = m.evalTranspose([2;3]);

a = IndexArray(struct('cells', [1;2;3]));
b = IndexArray(struct('cells', [3;1;2]));
m3 = TensorMap(); m3.fromTbl = a; m3.toTbl = b; m3.mergefds = {'cells'};
m3 = m3.setup();
out.perm = struct('v', m3.eval([10;20;30]), 'disp', m3.getDispatchInd());

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_tensormap.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_tensormap.mat\n');
