function dump_shale_fluid_catalog_reference(output_path)
% Direct MRST fixtures for both getShaleCompFluidCase catalog variants.

if nargin ~= 1
    error('Expected explicit output_path.');
end
this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(fileparts(this_file))));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add shale compositional

main_dir = fullfile(mrst_root, 'modules', 'shale', 'compositionalFns');
utils_dir = fullfile(mrst_root, 'modules', 'shale', 'utils');
main_cases = {'lumped_1', 'spe5', 'fourmix', '20components', ...
    'liquid_initial', 'vapor_initial', 'watertracer', 'simple', ...
    'verysimple', 'onlydecane', 'watermethane', 'onlymethane', 'c1c2', ...
    'barnett3comps', 'c1c2c3', 'bakken', 'eagleford', 'bakken_light'};
utility_cases = [main_cases, {'oil_1', 'oil_1_modified', ...
    'oil_1_modified_2', 'bakken_light_5comps', ...
    'barnett3comps_modified_2', 'barnett3comps_nat_vars', ...
    'barnett3comps_modified'}];

if contains(path, utils_dir)
    rmpath(utils_dir);
end
addpath(main_dir, '-begin'); clear getShaleCompFluidCase
main = collect_cases(main_cases);

rmpath(main_dir); addpath(main_dir); addpath(utils_dir, '-begin'); clear getShaleCompFluidCase
utility = collect_cases(utility_cases);
save(output_path, 'main', 'utility', '-v7');
fprintf('Wrote shale fluid catalog fixture to %s\n', output_path);
end

function out = collect_cases(cases)
ncase = numel(cases); maxcomp = 20;
out.case_names = cases;
out.component_names = cell(ncase, 1);
out.count = zeros(ncase, 1);
out.injection_count = zeros(ncase, 1);
out.injection = nan(ncase, maxcomp);
out.initial = nan(ncase, maxcomp);
out.pressure = nan(ncase, 1);
out.temperature = nan(ncase, 1);
out.Tcrit = nan(ncase, maxcomp);
out.Pcrit = nan(ncase, maxcomp);
out.Vcrit = nan(ncase, maxcomp);
out.acentric = nan(ncase, maxcomp);
out.molar_mass = nan(ncase, maxcomp);
out.bic = nan(ncase, maxcomp, maxcomp);
out.isotherm = nan(ncase, 2, maxcomp);
for i = 1:ncase
    [fluid, info] = getShaleCompFluidCase(cases{i});
    n = fluid.getNumberOfComponents();
    out.component_names{i} = strjoin(fluid.names, '|');
    out.count(i) = n;
    ni = numel(info.injection);
    out.injection_count(i) = ni;
    out.injection(i, 1:ni) = info.injection;
    out.initial(i, 1:n) = info.initial;
    out.pressure(i) = info.pressure;
    out.temperature(i) = info.temp;
    out.Tcrit(i, 1:n) = fluid.Tcrit;
    out.Pcrit(i, 1:n) = fluid.Pcrit;
    out.Vcrit(i, 1:n) = fluid.Vcrit;
    out.acentric(i, 1:n) = fluid.acentricFactors;
    out.molar_mass(i, 1:n) = fluid.molarMass;
    out.bic(i, 1:n, 1:n) = fluid.getBinaryInteraction();
    if isa(fluid, 'SorbedCompositionalFluid')
        out.isotherm(i, :, 1:n) = fluid.isotherm;
    end
end
end
