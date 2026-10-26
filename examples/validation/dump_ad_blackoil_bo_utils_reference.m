% Reference for the black-oil utilities getbO_BO, getbG_BO,
% getCapillaryPressureBO and getCellStatusVO, on SPE9's deck fluid
% (validate_ad_blackoil_bo_utils.py).
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat test-suite
[G, rock, fluid, deck] = setupSPE9();
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
p = linspace(50, 400, 12)'*barsa;
rs = linspace(0, 200, 12)';
isSat = mod((1:12)', 3) == 0;
bO_live = getbO_BO(model, p, rs, isSat);
bG_dry = getbG_BO(model, p, rs, isSat);
sW = linspace(0.12, 1, 12)';
sG = linspace(0, 0.8, 12)';
[pcOW, pcOG] = getCapillaryPressureBO(fluid, sW, sG);
[pcOW_empty, pcOG_empty] = getCapillaryPressureBO(fluid, [], []);
pcOW_empty = double(pcOW_empty); pcOG_empty = double(pcOG_empty);
% Cell status: water-only, oil-only, gas-only, both, and nothing, under all flags.
sO = [0; 0.7; 0; 0.3; 0; 1 - 1e-9; 0];
sWs = [1; 0.3; 0.2; 0.2; 1 - 1e-9; 0; 1 - 1e-7];
sGs = 1 - sO - sWs; sGs(sGs < 0) = 0;
status = zeros(numel(sO), 12);
k = 0;
for vapoil = [false, true]
    for disgas = [false, true]
        st = getCellStatusVO(sO, sWs, sGs, 'vapoil', vapoil, 'disgas', disgas);
        for j = 1:3
            k = k + 1;
            v = st{j};
            if isscalar(v), v = repmat(v, numel(sO), 1); end
            status(:, k) = v;
        end
    end
end
given = getCellStatusVO(sO, sWs, sGs, 'status', [1; 2; 3; 1; 2; 3; 0], ...
                        'vapoil', true, 'disgas', true);
given = double([given{:}]);
save(fullfile(here, 'ref_ad_blackoil_bo_utils.mat'), '-v7', 'p', 'rs', 'isSat', ...
     'bO_live', 'bG_dry', 'sW', 'sG', 'pcOW', 'pcOG', 'pcOW_empty', ...
     'pcOG_empty', 'sO', 'sWs', 'sGs', 'status', 'given');
