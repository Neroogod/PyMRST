% MRST/MATLAB references for batched eigensolvers and tensor assembly.
repo = '/Users/Shared/PyMRST_workspace';
utils = fullfile(repo, 'mrst-2024a-Geomech', 'utils');
addpath(utils);
addpath(fullfile(utils, 'tensorAssembly'));
addpath(fullfile(utils, 'tensorAssembly', 'utils'));

% multiEig and multiSymmEig are MEX build wrappers.  The public operation is
% a block-wise eig call, so use MATLAB's authoritative eig directly without
% compiling platform-specific binaries.
general_a = [0 -1; 1 0];
general_b = [3 1 0; 0 2 1; 0 0 4];
general_flat = [general_a(:); general_b(:)];
general_sizes = [2; 3];
general_values = [eig(general_a); eig(general_b)];

symm_a = [2 1; 1 2];
symm_b = [4 -1 0; -1 3 0.5; 0 0.5 2];
symm_flat = [symm_a(:); symm_b(:)];
symm_sizes = [2; 3];
[symm_va, symm_da] = eig(symm_a, 'vector');
[symm_vb, symm_db] = eig(symm_b, 'vector');
symm_values = [symm_da; symm_db];

t1 = IndexArray(struct('cells', [1;1;2], 'faces', [4;8;8]));
t2 = IndexArray(struct('faces', [4;8;8], 'nodes', [11;21;31]));

[projected, projection] = projIndexArray(t1, {'cells'});
projected_inds = projected.inds;
projection_inds = projection.inds;
sorted = sortIndexArray(t1, {'faces', 'cells'});
sorted_inds = sorted.inds;

renamed = replacefield(t1, {'cells', 'c'});
renamed_inds = renamed.inds;
renamed_names = renamed.fdnames;
interchanged = replacefield(t1, {'cells', 'faces', 'interchange'});
interchanged_inds = interchanged.inds;
interchanged_names = interchanged.fdnames;

disjoint = IndexArray(struct('cells', [3;4], 'faces', [9;10]));
concatenated = concatIndexArray(t1, disjoint, {'cells', 'faces'});
concatenated_inds = concatenated.inds;

% compIndexArray in MRST 2024a accesses an unset isvirtual property with ||,
% and its later comparison accidentally uses tbl1 twice. Preserve the actual
% failure as a regression fixture while PyMRST implements the documented intent.
try
    compIndexArray(t1, t1);
    comp_ran = true;
    comp_message = '';
catch err
    comp_ran = false;
    comp_message = err.message;
end

prod = TensorProd('tbl1', t1, 'tbl2', t2, 'reducefds', {'faces'});
prod = prod.setup();
left = [2; 3; 5]; right = [7; 11; 13];
product_out = prod.eval(left, right);
product_tbl3 = prod.tbl3.inds;
product_disp1 = prod.dispind1;
product_disp2 = prod.dispind2;
product_disp3 = prod.dispind3;

tensor = SparseTensor('prod', prod, 'value', left, ...
                      'argindex', 1, 'matlabsparse', true);
sparse_matrix = full(tensor.getMatrix());
sparse_applied = sparse_matrix*right;

destination = fullfile(repo, 'PyMRST', 'scripts', 'matlab', ...
                       'ref_utils_tensor_eig.mat');
save(destination, 'general_flat', 'general_sizes', 'general_values', ...
     'symm_flat', 'symm_sizes', 'symm_values', 'projected_inds', ...
     'projection_inds', 'sorted_inds', 'renamed_inds', 'renamed_names', ...
     'interchanged_inds', 'interchanged_names', 'concatenated_inds', ...
     'comp_ran', 'comp_message', ...
     'left', 'right', 'product_out', 'product_tbl3', 'product_disp1', ...
     'product_disp2', 'product_disp3', 'sparse_matrix', ...
     'sparse_applied', '-v7');
fprintf('wrote %s\n', destination);
