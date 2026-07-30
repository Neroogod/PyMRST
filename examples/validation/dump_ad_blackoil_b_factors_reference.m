%% Reference data for getbO_BO and getbG_BO dispatch behavior

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props deckformat

fn = fullfile(ROOTDIR, 'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');
deck = convertDeckUnits(readEclipseDeck(fn));
G = computeGeometry(initEclipseGrid(deck));
rock = compressRock(initEclipseRock(deck), G.cells.indexMap);
fluid = initDeckADIFluid(deck);
model = ThreePhaseBlackOilModel(G, rock, fluid);

p = [10; 20];
rs = [1; 2];
rv = [0.1; 0.2];
gas_present = logical([1; 0]);
oil_present = logical([0; 1]);

model.disgas = false;
model.fluid.bO = @(pressure) 2 + 0.1.*pressure;
bo_dead = getbO_BO(model, p, rs, gas_present);
model.disgas = true;
model.fluid.bO = @(pressure, ratio, saturated) ...
    2 + 0.1.*pressure + 0.2.*ratio + 0.3.*saturated;
bo_live = getbO_BO(model, p, rs, gas_present);

model.vapoil = false;
model.fluid.bG = @(pressure) 3 + 0.4.*pressure;
bg_dry = getbG_BO(model, p, rv, oil_present);
model.vapoil = true;
model.fluid.bG = @(pressure, ratio, liquid) ...
    3 + 0.4.*pressure + 0.5.*ratio + 0.6.*liquid;
bg_wet = getbG_BO(model, p, rv, oil_present);

save(fullfile(thisDir, 'ref_ad_blackoil_b_factors.mat'), '-v7', ...
     'p', 'rs', 'rv', 'gas_present', 'oil_present', ...
     'bo_dead', 'bo_live', 'bg_dry', 'bg_wet');
fprintf('Wrote ad-blackoil b-factor reference to %s\n', thisDir);
