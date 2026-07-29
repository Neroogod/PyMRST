%% Direct MRST reference for evaluateFunctionCellSubsetReg.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
x = (1:6)';
fn = {@(v) 2*v, @(v) 10 + v};
regions = [1; 2; 1; 2; 1; 2];
regions_offset = regions + 3;
v = evaluateFunctionCellSubsetReg([], fn, regions, x);
v_offset = evaluateFunctionCellSubsetReg([], fn, regions_offset, x);
save(fullfile(thisDir, 'ref_ad_core_region_subset.mat'), '-v7', ...
    'x', 'regions', 'regions_offset', 'v', 'v_offset');
fprintf('Wrote ref_ad_core_region_subset.mat\n');
