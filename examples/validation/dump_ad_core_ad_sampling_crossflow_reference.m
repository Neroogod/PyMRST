%% MRST reference for AD sampling, conversion, and cross-flow mixture density.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

sample = initVariablesADI([1; 2]);
converted = double2ADI([4; 5; 6], sample);
converted_value = converted.val;
converted_jacobian = full(converted.jac{1});
converted_double = double2ADI([7; 8], 1);

[sample_ad, is_ad] = getSampleAD([2], sample, [3]);
sample_ad_value = sample_ad.val;
[sample_numeric, is_numeric_ad] = getSampleAD([2], [3]);

properties = struct('a', [1], 'b', []);
evaluated_a = structPropEvaluated(properties, 'a');
evaluated_b = structPropEvaluated(properties, 'b');

map = struct();
map.perf2well = [1; 1; 2];
map.W = {1, 2};
mass_flux = [-2, -4; 1, 2; 0, 0];
volume_flux = [-1; 2; 3];
surface_mass_flux = [10, 20; 0, 0];
mixture = crossFlowMixtureDensity( ...
    mass_flux, volume_flux, surface_mass_flux, map);

save(fullfile(thisDir, 'ref_ad_core_ad_sampling_crossflow.mat'), '-v7', ...
    'converted_value', 'converted_jacobian', 'converted_double', ...
    'sample_ad_value', 'is_ad', 'sample_numeric', 'is_numeric_ad', ...
    'evaluated_a', 'evaluated_b', 'mixture');
fprintf('Wrote ref_ad_core_ad_sampling_crossflow.mat\n');
