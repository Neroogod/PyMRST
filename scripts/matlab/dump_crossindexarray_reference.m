% Reference for validate_cross_index_array.py
out = struct();

t1 = IndexArray(struct('cells', [1;1;2], 'faces', [4;8;8]));
t2 = IndexArray(struct('faces', [4;8;8], 'nodes', [11;21;31]));
[c, s] = crossIndexArray(t1, t2, {'faces'});
out.join = struct('inds', c.inds, 'fdnames', {c.fdnames}, ...
                  'i1', s{1}.inds(:), 'i2', s{2}.inds(:));

a = IndexArray(struct('cells', [1;1;2], 'faces', [4;8;8]));
b = IndexArray(struct('cells', [1;2], 'faces', [8;8], 'v', [6;7]));
c2 = crossIndexArray(a, b, {'cells', 'faces'});
out.two = struct('inds', c2.inds, 'fdnames', {c2.fdnames});

d = IndexArray(struct('faces', [2;3]));
e = IndexArray(struct('faces', [3;4], 'nodes', [8;9]));
c3 = crossIndexArray(d, e, {'faces'});
out.partial = struct('inds', c3.inds, 'fdnames', {c3.fdnames});

p1 = IndexArray(struct('cells', [1;2]));
p2 = IndexArray(struct('nodes', [8;9;10]));
c4 = crossIndexArray(p1, p2, {}, 'optpureproduct', true);
out.product = struct('inds', c4.inds, 'fdnames', {c4.fdnames});

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_crossindexarray.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_crossindexarray.mat\n');
