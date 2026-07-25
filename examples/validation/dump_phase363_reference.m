%% Dump Phase 363 reference: 1D Buckley-Leverett AD profile, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p363_bl.mat.
%
% Ports MRST's introductory example `adBuckleyLeverett1D`: an incompressible 1D
% waterflood driven by a flux BC at the inlet and a pressure BC at the outlet,
% stepped for one pore volume injected. This dumps the final water-saturation
% profile so the PyMRST example reproduces it cell for cell.
%
% The fluid viscosities/densities are set explicitly (unit ratio, equal density)
% so the horizontal displacement is the pure Buckley-Leverett problem and the
% two codes share identical inputs.

mrstModule add ad-blackoil ad-core ad-props

gravity reset off;
thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([50, 1, 1], [1000, 10, 10]*meter));
rock = makeRock(G, 1*darcy, 0.3);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 1]*centi*poise, ...
                           'rho', [1000, 1000], 'c', [0, 0]);
model = TwoPhaseOilWaterModel(G, rock, fluid);

state0 = initResSol(G, 50*barsa, [0, 1]);

pv = poreVolume(G, rock);
injRate = sum(pv)/(500*day);
bc = fluxside([], G, 'xmin', injRate, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', 0*barsa, 'sat', [0, 1]);

model = model.validateModel();
state0 = model.validateState(state0);

n  = 25;
dT = (500/n)*day;
solver = NonLinearSolver();
state = state0;
for i = 1:n
    state = solver.solveTimestep(state, dT, model, 'bc', bc);
end

sW = state.s(:, 1);

save(fullfile(thisDir, 'ref_p363_bl.mat'), '-v7', 'sW', 'n', 'injRate');

fprintf('Wrote ref_p363_bl.mat to %s\n', thisDir);
fprintf('  sW inlet = %.4f, outlet = %.4f, swept vol = %.3f\n', ...
        sW(1), sW(end), sum(sW));
