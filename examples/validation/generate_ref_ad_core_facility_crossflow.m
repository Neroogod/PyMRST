% Generate MRST reference values for crossFlowMixture/calculatePhaseRate.
mrstModule add ad-core

W(1).compi = [0.8, 0.2];
W(2).compi = [0.1, 0.9];
map.W = W;
map.perf2well = [1; 1; 2; 2];
map.isInjector = [true; false];
map.perforationSum = sparse(map.perf2well, 1:4, 1, 2, 4);

flux = [-4, -2; 3, 12; -12, -8; 10, 5];
topside = vertcat(W.compi);
mixture = crossFlowMixture(flux, topside, map);
mixture_conservative = crossFlowMixture(flux, topside, map, true);

tdp = [-2; 3; -4; 5];
mobility = {[2; 1; 3; 2], [1; 4; 2; 1]};
model.verbose = 0;
phase_rate = calculatePhaseRate(tdp, mobility, map, true, model);
phase_rate_no_crossflow = calculatePhaseRate( ...
    tdp, mobility, map, false, model);
phase_rate = horzcat(phase_rate{:});
phase_rate_no_crossflow = horzcat(phase_rate_no_crossflow{:});

out = fullfile(fileparts(mfilename('fullpath')), ...
    'ref_ad_core_facility_crossflow.mat');
save(out, 'flux', 'topside', 'mixture', 'mixture_conservative', ...
    'tdp', 'phase_rate', 'phase_rate_no_crossflow');
