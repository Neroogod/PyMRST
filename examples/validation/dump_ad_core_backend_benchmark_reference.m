function dump_ad_core_backend_benchmark_reference(output_file)
% Independent MRST reference for PyMRST benchmarkAutoDiffBackends expressions.

if nargin < 1
    output_file = fullfile(tempdir, 'ad_core_backend_benchmark_reference.mat');
end

mrstModule add ad-core
rng(0);
n = 17;
values = cell(1, 4);
for i = 1:numel(values)
    values{i} = rand(n, 1);
end
[x, y, z, unused] = initVariablesADI(values{:}); %#ok<ASGLU>
subset = (1:floor(n/7):n)';

outputs = {
    x.*y, ...
    x.*values{2}, ...
    x.*y + 2.*z, ...
    x.^2 + 0.5.*y.*z, ...
    subsref(x.*y + z, substruct('()', {subset}))
};
names = {'cell_xy', 'cell_xv', 'cell_xy_2z', 'cell_polynomial', 'subset'};

reference = struct();
for i = 1:numel(outputs)
    blocks = outputs{i}.jac;
    jacobian = full(horzcat(blocks{:}));
    reference.(names{i}) = struct( ...
        'value', value(outputs{i}), ...
        'jacobian', jacobian);
end
save(output_file, 'reference', 'values', 'subset');
fprintf('Wrote %s\n', output_file);
end
