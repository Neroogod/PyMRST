function run_mrst_victory_flash_reference(input_path, output_path)
% Run canonical MRST standaloneFlash for the Victory C1/C3/C14 comparison.
%
% Example:
%   run_mrst_victory_flash_reference('results/run/inputs.mat', ...
%                                    'results/run/mrst_results.mat')

if nargin ~= 2
    error('Expected explicit input_path and output_path.');
end
input_path = char(input_path);
output_path = char(output_path);
if ~isfile(input_path)
    error('Input fixture does not exist: %s', input_path);
end

this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(this_file)));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
startup_file = fullfile(mrst_root, 'startup.m');
if ~isfile(startup_file)
    error('MRST startup file not found: %s', startup_file);
end
run(startup_file);
mrstModule add ad-core compositional

in = load(input_path);
required = {'p', 'T', 'z', 'Tcrit', 'Pcrit', 'Vcrit', 'acentric', ...
            'molar_mass', 'bic', 'input_digest'};
for i = 1:numel(required)
    if ~isfield(in, required{i})
        error('Input fixture is missing field %s.', required{i});
    end
end
assert(size(in.z, 2) == 3, 'Expected N-by-3 C1/C3/C14 compositions.');
assert(numel(in.p) == size(in.z, 1) && numel(in.T) == size(in.z, 1), ...
       'Pressure, temperature and composition row counts must match.');

names = {'Methane', 'n-Propane', 'n-Tetradecane'};
mixture = CompositionalMixture(names, reshape(in.Tcrit, 1, []), ...
    reshape(in.Pcrit, 1, []), reshape(in.Vcrit, 1, []), ...
    reshape(in.acentric, 1, []), reshape(in.molar_mass, 1, []));
mixture = mixture.setBinaryInteraction(in.bic);
eos = EquationOfStateModel([], mixture, 'Peng-Robinson');

% Warm up MATLAB dispatch/JIT separately from the measured batch.
standaloneFlash(in.p(1), in.T(1), in.z(1, :), eos);
tic;
[L, x, y, Z_L, Z_V, rhoL, rhoV, reports] = ...
    standaloneFlash(in.p(:), in.T(:), in.z, eos);
flash_elapsed_seconds = toc;

input_digest = char(in.input_digest);
schema_version = 'victory-flash-comparison-v1';
matlab_version = version;
mrst_source_root = mrst_root;
component_names = names;
save(output_path, 'L', 'x', 'y', 'Z_L', 'Z_V', 'rhoL', 'rhoV', ...
     'reports', 'flash_elapsed_seconds', 'input_digest', 'schema_version', ...
     'matlab_version', 'mrst_source_root', 'component_names', '-v7');
fprintf('Wrote MRST flash comparison (%d rows, %.6g s) to %s\n', ...
        numel(L), flash_elapsed_seconds, output_path);
end
