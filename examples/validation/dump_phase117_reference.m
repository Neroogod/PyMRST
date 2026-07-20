function dump_phase117_reference()
% Phase 117 reference: compositional performance benchmark structure.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add compositional ad-core linearsolvers ad-props ad-blackoil

dims = [50, 50, 50];
[fluid, info] = getBenchmarkMixture('spe5');
ncomp = fluid.getNumberOfComponents();
cells = prod(dims);
degrees_of_freedom = cells*ncomp;

sparse_backend = SparseAutoDiffBackend();
diagonal_backend = DiagonalAutoDiffBackend();
mex_backend = DiagonalAutoDiffBackend('useMex', true);
row_backend = DiagonalAutoDiffBackend('useMex', true, ...
    'rowMajor', true, 'deferredAssembly', true);

backend_labels = {'Sparse'; 'Diagonal'; 'Diagonal (MEX)'; 'Diagonal (MEX+RowMajor)'};
backend_use_mex = [false; diagonal_backend.useMex; mex_backend.useMex; row_backend.useMex];
backend_row_major = [false; diagonal_backend.rowMajor; mex_backend.rowMajor; row_backend.rowMajor];
backend_deferred = [false; diagonal_backend.deferredAssembly; mex_backend.deferredAssembly; row_backend.deferredAssembly];
initial = info.initial(:);
injection = info.injection(:);
pressure = info.pressure;
temperature = info.temp;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p117_compositional_performance.mat');
save(out, 'dims', 'cells', 'ncomp', 'degrees_of_freedom', ...
    'backend_labels', 'backend_use_mex', 'backend_row_major', ...
    'backend_deferred', 'initial', 'injection', 'pressure', 'temperature');
fprintf('Wrote ref_p117_compositional_performance.mat to %s\n', fileparts(out));
end
