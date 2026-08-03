function dump_poro_params_basis_reference(output_file)
%DUMP_PORO_PARAMS_BASIS_REFERENCE Certify every isotropic elastic basis.
%
% The two E/Kv bases are intentionally included.  They are algebraically
% ambiguous; this fixture records the branch selected by MRST's fzero loop.

if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_poro_params_basis.mat');
end

mrstModule add ad-mechanics
warning('off', 'all');

phi = 0.22;
base = poroParams(phi, false, ...
    'E', 12e9, 'nu', 0.25, 'alpha', 0.8, 'M', 5e9);

drained_names = {'K', 'E', 'G', 'lambda', 'nu', 'K_v'};
undrained_names = {'K_u', 'E_u', 'G', 'lambda_u', 'nu_u', 'K_vu'};
drained = make_cases(phi, base, drained_names);
undrained = make_cases(phi, base, undrained_names);

save(output_file, 'phi', 'base', 'drained_names', 'undrained_names', ...
    'drained', 'undrained', '-v7');
fprintf('Wrote %s (%d drained + %d undrained bases)\n', output_file, ...
    numel(drained), numel(undrained));
end

function cases = make_cases(phi, base, names)
pairs = nchoosek(1:numel(names), 2);
cases = cell(size(pairs, 1), 1);
for i = 1:size(pairs, 1)
    left = names{pairs(i, 1)};
    right = names{pairs(i, 2)};
    [result, history, residuals] = poroParams(phi, false, ...
        left, base.(left), right, base.(right));
    item.left = left;
    item.right = right;
    item.result = result;
    item.history = history;
    item.residuals = residuals;
    cases{i} = item;
end
end
