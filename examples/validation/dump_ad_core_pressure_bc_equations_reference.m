%% MRST reference for pressure boundary contributions and equation assembly.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
gravity reset
gravity([0, 0, 10]);
gravity on

G = struct();
G.faces = struct( ...
    'neighbors', [0, 1; 1, 2; 2, 0], ...
    'centroids', [0, 0, 0; 0, 0, 1.5; 0, 0, 3], ...
    'num', 3);
G.cells = struct( ...
    'centroids', [0, 0, 1; 0, 0, 2], ...
    'num', 2);
s = struct('T_all', [2; 4; 3]);
bc = struct( ...
    'face', [1; 3], ...
    'type', {{'pressure'; 'pressure'}}, ...
    'value', [120; 80]);
bc_ad = bc2ADbc(G, bc);

p = {[100; 100]};
rho = {[2; 3]};
mob = {[0.5; 0.25]};
b = {[1.2; 1.5]};
[legacy, legacy_cells] = pressureBCContrib(G, s, p, rho, mob, b, bc);
legacy_value = legacy{1};

p_ad = {initVariablesADI([100; 100])};
[adi, adi_cells] = pressureBCContribADI(G, s, p_ad, rho, mob, b, bc_ad);
adi_value = adi{1}.val;
adi_jacobian = full(adi{1}.jac{1});

numeric_combined = combineEquations([1; 2], 3);
[pressure_ad, bhp_ad] = initVariablesADI([1; 2], 5);
ad_combined = combineEquations(3*pressure_ad + bhp_ad, 2*bhp_ad);
combined_value = ad_combined.val;
combined_jacobian = full(ad_combined.jac{1});

save(fullfile(thisDir, 'ref_ad_core_pressure_bc_equations.mat'), '-v7', ...
    'legacy_value', 'legacy_cells', 'adi_value', 'adi_cells', ...
    'adi_jacobian', 'numeric_combined', 'combined_value', ...
    'combined_jacobian');
fprintf('Wrote ref_ad_core_pressure_bc_equations.mat\n');
gravity reset
