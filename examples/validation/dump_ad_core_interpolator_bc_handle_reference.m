%% MRST reference for multidimensional interpolation, BC maps, and HandleStruct.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

x = [2; 1; 0];
y = [0; 3];
[xx, yy] = ndgrid(x, y);
table = 2*xx + 3*yy;
[fn, ~, epsilons] = getMultiDimInterpolator({x, y}, table);
numeric_value = fn([0.5; 1.5], [1; 2]);
x_ad = initVariablesADI([0.5; 1.5]);
ad_value = fn(x_ad, [1; 2]);
ad_jacobian = full(ad_value.jac{1});
ad_value = ad_value.val;

G = struct();
G.faces = struct('neighbors', [0, 1; 1, 2; 2, 0], 'num', 3);
G.cells = struct('num', 2);
bc = struct('face', [1; 3], 'value', [10; 20]);
bc_ad = bc2ADbc(G, bc);
cell2bcface = full(bc_ad.cell2bcface);
bcface2cell = full(bc_ad.bcface2cell);

wrapped_ad = initVariablesADI([1; 2]);
wrapped = HandleStruct(struct( ...
    'ad', wrapped_ad, 'parts', {{[3], [4, 5]}}));
reduced = wrapped.reduce();
reduced_ad = reduced.ad;
reduced_parts = reduced.parts;
evaluated_ad = wrapped.structPropEvaluated('ad');

save(fullfile(thisDir, 'ref_ad_core_interpolator_bc_handle.mat'), '-v7', ...
    'epsilons', 'numeric_value', 'ad_value', 'ad_jacobian', ...
    'cell2bcface', 'bcface2cell', 'reduced_ad', 'reduced_parts', ...
    'evaluated_ad');
fprintf('Wrote ref_ad_core_interpolator_bc_handle.mat\n');
