%% Reference for black-oil phase flux/property utility functions

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset; gravity off

G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
rock = makeRock(G, 1, 0.2);
fluid = initSimpleADIFluid('phases', 'WOG', ...
    'mu', [1, 1, 1], 'rho', [1000, 700, 2]);
fluid.pcOW = @(sw) 2.*sw;
fluid.pcOG = @(sg) 3.*sg;
fluid.bW = @(p) 1 + 0.01.*p;
fluid.muW = @(p) 2 + 0.001.*p;
fluid.bO = @(p, rs, sat) 1.2 + 0.02.*p + 0.03.*rs + 0.04.*sat;
fluid.muO = @(p, rs, sat) 3 + 0.002.*p + 0.01.*rs;
fluid.bG = @(p, rv, sat) 4 + 0.04.*p + 0.05.*rv + 0.06.*sat;
fluid.muG = @(p, rv, sat) 0.5 + 0.003.*p + 0.02.*rv;
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true, 'vapoil', true);
model.operators = setupOperatorsTPFA(G, rock);

p = [10; 12; 11]; sw = [0.2; 0.3; 0.25]; sg = [0.1; 0.15; 0.2];
so = 1-sw-sg; rs = [2; 2.5; 3]; rv = [0.01; 0.02; 0.03];
krw = sw; kro = so; krg = sg;
T = model.operators.T; gdz = [0.1; -0.2];
gas_sat = logical([1; 0; 1]); oil_sat = logical([0; 1; 1]);

[vW,bW,mobW,rhoW,pW,upW,dpW,muW] = ...
    getFluxAndPropsWater_BO(model,p,sw,krw,T,gdz);
[vO,bO,mobO,rhoO,pO,upO,dpO,muO] = ...
    getFluxAndPropsOil_BO(model,p,so,kro,T,gdz,rs,gas_sat);
[vG,bG,mobG,rhoG,pG,upG,dpG,muG] = ...
    getFluxAndPropsGas_BO(model,p,sg,krg,T,gdz,rv,oil_sat);

save(fullfile(thisDir, 'ref_ad_blackoil_phase_flux.mat'), '-v7', ...
     'p','sw','sg','so','rs','rv','krw','kro','krg','T','gdz', ...
     'gas_sat','oil_sat', ...
     'vW','bW','mobW','rhoW','pW','upW','dpW','muW', ...
     'vO','bO','mobO','rhoO','pO','upO','dpO','muO', ...
     'vG','bG','mobG','rhoG','pG','upG','dpG','muG');
fprintf('Wrote ad-blackoil phase-flux reference to %s\n', thisDir);
