function dump_poro_params_reference(output_file)
% Generate representative MRST ad-mechanics poroParams results.

if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_poro_params.mat');
end

mrstModule add ad-mechanics

cases = cell(3, 1);
cases{1} = poroParams(0.20, false, ...
    'E', 12e9, 'nu', 0.25, 'alpha', 0.8, 'M', 5e9);
cases{2} = poroParams(0.25, false, ...
    'K', 9e9, 'G', 4e9, 'alpha', 0.75, 'M', 6e9);
cases{3} = poroParams(0.20, true, ...
    'K', 8e9, 'G', 4e9, 'alpha', 0.8, 'M', 5e9);

save(output_file, 'cases', '-v7');
fprintf('Wrote %s\n', output_file);
end
