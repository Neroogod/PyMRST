function dump_ad_core_regional_relperm_reference(output_file)
% MRST AD reference for differentRegionFunctionsExample regional curves.

if nargin < 1
    output_file = fullfile(tempdir, 'ad_core_regional_relperm_reference.mat');
end
mrstModule add ad-core

saturation = [0.2; 0.4; 0.6; 0.8];
regions = [1; 1; 2; 2];
s = initVariablesADI(saturation);
so = 1 - s;
water = 0*s;
oil = 0*s;
quadratic = regions == 1;
linear = regions == 2;
water(quadratic) = s(quadratic).^2;
water(linear) = s(linear);
oil(quadratic) = so(quadratic).^2;
oil(linear) = so(linear);

reference = struct( ...
    'water', value(water), ...
    'oil', value(oil), ...
    'water_jacobian', full(water.jac{1}), ...
    'oil_jacobian', full(oil.jac{1}));
save(output_file, 'reference', 'saturation', 'regions');
fprintf('Wrote %s\n', output_file);
end
