%% MRST reference for facility property packing and nozzle valve.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

rho_n = [800; 900];
D = 0.1;
coeff = 0.8;
velocity = [2; -3];
area = pi*(D/2).^2;
dp_velocity = nozzleValve(velocity, rho_n, D, coeff, 'velocity');
dp_volume = nozzleValve(velocity.*area, rho_n, D, coeff, 'volumeRate');
dp_mass = nozzleValve(velocity.*rho_n.*area, rho_n, D, coeff);

W = struct('cells', [2; 4]);
p = [100; 110; 120; 130];
mob = {[1; 2; 3; 4], [5; 6; 7; 8]};
rho = {[700; 710; 720; 730]};
dissolved = {{[0.1; 0.2; 0.3; 0.4]}, []};
comp = {[0.9; 0.8; 0.7; 0.6]};
wellvars = {[10; 20], [30; 40; 50]};
wellvarnames = {'a', 'b'};
varmaps = {[1; 2], [2; 1; 2]};
wellmap = [1, 2; 2, 0];
packed = packPerforationProperties(W, p, mob, rho, dissolved, comp, ...
    wellvars, wellvarnames, varmaps, wellmap, 1);
[p_out, mob_out, rho_out, dissolved_out, comp_out, wellvars_out] = ...
    unpackPerforationProperties(packed);
mob_1 = mob_out{1}; mob_2 = mob_out{2};
rho_1 = rho_out{1};
dissolved_1 = dissolved_out{1}{1};
comp_1 = comp_out{1};
extra_1 = wellvars_out{1}; extra_2 = wellvars_out{2};

save(fullfile(thisDir, 'ref_ad_core_facility_helpers.mat'), '-v7', ...
    'dp_velocity', 'dp_volume', 'dp_mass', 'p_out', 'mob_1', 'mob_2', ...
    'rho_1', 'dissolved_1', 'comp_1', 'extra_1', 'extra_2');
fprintf('Wrote ref_ad_core_facility_helpers.mat\n');
