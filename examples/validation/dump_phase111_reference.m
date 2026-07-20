function dump_phase111_reference()
% Phase 111 reference: SPE1 convertRSRVtoKValue tables.

mrstModule add ad-core ad-blackoil compositional ad-props test-suite

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
mrstDataDirectory(fullfile(mrstRoot, 'examples', 'data'));
[G, rock, fluid, deck, state] = setupSPE1(); %#ok<ASGLU>

model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
props = convertRSRVtoKValue(model, 'np', 12, 'nz', 16, ...
    'interpolation', 'linear', 'minPressure', 300*barsa, ...
    'maxPressure', 500*barsa);

k_K_o = props.k_values.K_o;
k_K_g = props.k_values.K_g;
k_freeGas = props.k_values.freeGas;
k_z_g = props.k_values.z_g;
k_z_o = props.k_values.z_o;
k_pressure = props.k_values.pressure;

p_rhoO = props.properties.rhoO;
p_rhoG = props.properties.rhoG;
p_muO = props.properties.muO;
p_muG = props.properties.muG;
p_pressure = props.properties.pressure;
p_x_o = props.properties.x_o;
p_y_g = props.properties.y_g;

[P0, RS0, SG0, direct] = localSampleCloud(model, 12, 16, 300*barsa, 500*barsa);
cloud_p = P0(:);
cloud_rs = RS0(:);
cloud_sG = SG0(:);
cloud_zg = direct.zg;
cloud_xg = direct.xg;
cloud_Kg = direct.Kg;
cloud_rhoO = direct.rhoO;
cloud_muO = direct.muO;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p111_spe1_kvalue.mat');
save(out, 'k_K_o', 'k_K_g', 'k_freeGas', 'k_z_g', 'k_z_o', 'k_pressure', ...
    'p_rhoO', 'p_rhoG', 'p_muO', 'p_muG', 'p_pressure', 'p_x_o', 'p_y_g', ...
    'cloud_p', 'cloud_rs', 'cloud_sG', 'cloud_zg', 'cloud_xg', 'cloud_Kg', ...
    'cloud_rhoO', 'cloud_muO');
fprintf('Wrote ref_p111_spe1_kvalue.mat to %s\n', fileparts(out));
fprintf('  SPE1 K-value tables: %d x %d\n', size(k_K_o, 1), size(k_K_o, 2));
end

function [P, RS, SG, direct] = localSampleCloud(model, np, nz, minPressure, maxPressure)
    fluid = model.fluid;
    p0 = linspace(minPressure, maxPressure, np)';
    if model.disgas
        rs0 = fluid.rsSat(p0);
        rs0 = linspace(1, 2*max(rs0), nz);
    else
        rs0 = linspace(1, 100, nz);
    end

    [P, RS] = meshgrid(p0, rs0);
    SG = 0*RS;
    for pNo = 1:numel(p0)
        if model.disgas
            rssat = fluid.rsSat(P(1, pNo));
        else
            rssat = 0;
        end
        act = RS(:, pNo) > rssat;
        RS(act, pNo) = rssat;
        SG(act, pNo) = linspace(0, 1, nnz(act));
    end
    p = P(:);
    sG = SG(:);
    rs = RS(:);
    sO = 1 - sG;
    [xo, xg, yo, yg, zo, zg, rhoO, rhoG, muO, muG, isSat] = blackOilToMassFraction(model, p, sO, sG, rs, nan); %#ok<ASGLU>
    direct.zg = zg;
    direct.xg = xg;
    direct.Kg = yg./xg;
    direct.rhoO = rhoO;
    direct.muO = muO;
end
