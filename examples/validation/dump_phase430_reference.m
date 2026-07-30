%% Dump Phase 430 reference: obj2hash, from MRST.

mrstModule add ad-core

thisDir = fileparts(mfilename('fullpath'));

numeric = [1, 2; 3, 4];
logical_value = logical([1, 0, 1]);
char_value = 'PyMRST';
cell_value = {'yes', logical([1, 0]), [2.5; -7]};
struct_value = struct('numeric', numeric, 'cell', {cell_value}, ...
                      'ignored', 99);
sparse_value = sparse([1, 3, 2], [1, 2, 4], [2.5, -1, 7], 3, 4);

out_numeric = obj2hash(numeric);
out_logical = obj2hash(logical_value);
out_char = obj2hash(char_value);
out_cell = obj2hash(cell_value);
[out_struct, hs] = obj2hash(struct_value);
out_struct_numeric = hs.numeric;
out_struct_cell = hs.cell;
out_skip = obj2hash(struct_value, 'skip', {'ignored'});
out_max_size = obj2hash(numeric, 'maxSize', 1);
out_empty = obj2hash([]);
out_sparse = obj2hash(sparse_value);

save(fullfile(thisDir, 'ref_p430_obj2hash.mat'), '-v7', ...
     'numeric', 'logical_value', 'char_value', ...
     'out_numeric', 'out_logical', 'out_char', 'out_cell', ...
     'out_struct', 'out_struct_numeric', 'out_struct_cell', ...
     'out_skip', 'out_max_size', 'out_empty', 'out_sparse');
fprintf('Wrote ref_p430_obj2hash.mat to %s\n', thisDir);
