%% Dump Phase 425 reference: degree-0 DG transport (Buckley-Leverett), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p425_dg0_bl.mat.
%
% Degree-0 TransportModelDG is plain finite-volume upwind (no slope limiter), so
% PyMRST's DGScalarTransport at degree 0 must reproduce it dof-for-dof on this
% incompressible Buckley-Leverett displacement: 30x1 grid, quadratic relperm,
% equal viscosity (f_w = s^2/(s^2+(1-s)^2)), constant total flux 1, water
% injected at the left, 30 steps of dt=1 (one pore volume).

mrstModule add dg ad-core ad-props ad-blackoil sequential vem vemmech

thisDir = fileparts(mfilename('fullpath'));

G    = computeGeometry(cartGrid([30, 1]));
G    = computeCellDimensions(G);
rock = makeRock(G, 1, 1);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 1], 'rho', [1, 1]);
model  = GenericBlackOilModel(G, rock, fluid, 'gas', false);
state0 = initResSol(G, 1, [0, 1]);
state0.flux(1:G.cells.num + 1) = 1;
bc = fluxside([], G, 'left',   1, 'sat', [1, 0]);
bc = fluxside(bc, G, 'right', -1, 'sat', [1, 0]);
schedule = simpleSchedule(ones(30, 1), 'bc', bc);

tmodelDG0 = TransportModelDG(model, 'degree', 0);
[~, stDG0, ~] = simulateScheduleAD(state0, tmodelDG0, schedule);

% Water saturation per cell at each report step.
nstep = numel(stDG0);
sw = zeros(G.cells.num, nstep);
for i = 1:nstep
    sw(:, i) = stDG0{i}.s(:, 1);
end

save(fullfile(thisDir, 'ref_p425_dg0_bl.mat'), '-v7', 'sw');
fprintf('Wrote ref_p425_dg0_bl.mat (%d cells, %d steps)\n', G.cells.num, nstep);
