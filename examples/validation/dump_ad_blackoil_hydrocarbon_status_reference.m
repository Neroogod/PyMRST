%% Reference for calculateHydrocarbonsFromStatusBO

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil

fluid.rsSat = @(p) 10 + 0.1.*p;
fluid.rvSat = @(p) 0.2 + 0.01.*p;
encoded = [1; 2; 3];
status = {encoded == 1, encoded == 2, encoded == 3};
oil_space = [0.8; 0.7; 0.6];
x = [7; 0.9; 0.25];
rs_input = [1; 2; 3];
rv_input = [0.1; 0.2; 0.3];
pressure = [100; 200; 300];

[sG, rs, rv, rsSat, rvSat] = calculateHydrocarbonsFromStatusBO( ...
    fluid, status, oil_space, x, rs_input, rv_input, pressure, true, true);
[sG_dead, rs_dead, rv_dead, rsSat_dead, rvSat_dead] = ...
    calculateHydrocarbonsFromStatusBO( ...
        fluid, status, oil_space, x, rs_input, rv_input, pressure, false, false);

save(fullfile(thisDir, 'ref_ad_blackoil_hydrocarbon_status.mat'), '-v7', ...
     'encoded', 'oil_space', 'x', 'rs_input', 'rv_input', 'pressure', ...
     'sG', 'rs', 'rv', 'rsSat', 'rvSat', ...
     'sG_dead', 'rs_dead', 'rv_dead', 'rsSat_dead', 'rvSat_dead');
fprintf('Wrote ad-blackoil hydrocarbon-status reference to %s\n', thisDir);
