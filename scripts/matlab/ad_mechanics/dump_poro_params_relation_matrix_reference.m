function dump_poro_params_relation_matrix_reference(output_file)
% Exhaustively invert every argument of every poroParams relation.
if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_poro_params_relation_matrix.mat');
end
mrstModule add ad-mechanics
warning('off', 'all');
phi = 0.23;
[base, ~, ~, eqs] = poroParams(phi, true, ...
    'E', 11e9, 'nu', 0.24, 'alpha', 0.82, 'K_f', 2.1e9);
cases = {};
for relation_index = 1:numel(eqs)
    args = eqs(relation_index).args;
    for unknown_index = 1:numel(args)
        unknown = args{unknown_index};
        options = {};
        inputs = struct();
        for j = 1:numel(args)
            if j == unknown_index, continue; end
            name = args{j};
            value = base.(name);
            options = [options, {name, value}]; %#ok<AGROW>
            inputs.(name) = value;
        end
        item.relation_index = relation_index;
        item.unknown = unknown;
        item.inputs = inputs;
        try
            [item.result, item.history, item.residuals] = ...
                poroParams(phi, true, options{:});
            item.error = '';
        catch exception
            item.result = struct();
            item.history = {};
            item.residuals = [];
            item.error = exception.message;
        end
        cases{end + 1, 1} = item; %#ok<AGROW>
    end
end
mrst_version = 'mrst-2024a-Geomech';
matlab_version = version();
save(output_file, 'phi', 'base', 'cases', 'mrst_version', ...
    'matlab_version', '-v7');
fprintf('Wrote %s (%d relation inversions)\n', output_file, numel(cases));
end
