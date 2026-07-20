% Phase 15 reference: dual porosity (MRST dual-porosity/)
%
% Two cases, both driven by pressure boundary conditions as MRST's own
% examples are:
%
%   1ph  a depleting fractured column (MRST's example_1ph): the fracture
%        drains fast, the matrix bleeds into it slowly.
%   2ph  capillary imbibition (MRST's example_2ph_imbibition, but with 20 m
%        matrix blocks and a half-day step, so the matrix visibly lags the
%        fracture front instead of equilibrating inside one step).

mrstModule add ad-core ad-blackoil ad-props dual-porosity

here = fileparts(mfilename('fullpath'));
ref = struct();

%% ---------------------------------------------------------------- 1 phase
G  = computeGeometry(cartGrid([100, 1, 1], [60, 1, 1]*meter));
rock_matrix   = makeRock(G, 1*darcy,    0.3);
rock_fracture = makeRock(G, 1000*darcy, 0.01);

fluid_matrix   = initSimpleADIFluid('phases', 'W', 'c', 1e-06);
fluid_fracture = initSimpleADIFluid('phases', 'W', 'c', 1e-06);

model = WaterModelDP(G, {rock_fracture, rock_matrix}, ...
                        {fluid_fracture, fluid_matrix});
model.transfer_model_object = KazemiSinglePhaseTransferFunction( ...
    'KazemiShapeFactor', [10, 10, 10]);
model = model.validateModel();

% MRST's default convergence (toleranceCNV = 1e-3) stops the Newton loop far
% short of the true solution, which shows up as a ~1e-5 difference against a
% tightly converged Python solve. Tighten it so the reference is the answer
% to the equations rather than the answer to the stopping rule.
model.toleranceCNV = 1e-12;
model.toleranceMB  = 1e-14;
model.nonlinearTolerance = 1e-12;

state = initResSol(G, 1000*psia);
state.wellSol = initWellSolAD([], model, state);
state.pressure_matrix = state.pressure;

bc = pside([], G, 'xmin', 0*psia, 'sat', 1);
solver = NonLinearSolver('maxIterations', 100);

dt1 = 0.1*day;
n1  = 20;
p1  = zeros(G.cells.num, n1);
pm1 = zeros(G.cells.num, n1);
for k = 1:n1
    state = solver.solveTimestep(state, dt1, model, 'bc', bc);
    p1(:, k)  = state.pressure;
    pm1(:, k) = state.pressure_matrix;
end

ref.n1        = n1;
ref.dt1       = dt1;
ref.p1        = p1;
ref.pm1       = pm1;
ref.perm_m1   = rock_matrix.perm;
ref.perm_f1   = rock_fracture.perm;
ref.poro_m1   = rock_matrix.poro;
ref.poro_f1   = rock_fracture.poro;

fprintf('1ph: fracture %.1f -> %.1f psia, matrix %.1f psia at the outlet\n', ...
        1000, p1(1, end)/psia, pm1(1, end)/psia);

%% ---------------------------------------------------------------- 2 phase
G2 = computeGeometry(cartGrid([30, 1, 1], [30, 1, 1]*meter));
rock_matrix2   = makeRock(G2, 10*milli*darcy,  0.1);
rock_fracture2 = makeRock(G2, 100*milli*darcy, 0.01);

Pe = 7*barsa;
fm2 = initSimpleADIFluid('phases', 'WO', 'mu', [1, 1]*centi*poise, ...
                         'rho', [1000, 800], 'n', [1, 1], ...
                         'c', [1e-12; 1e-12], 'pRef', 0);
ff2 = initSimpleADIFluid('phases', 'WO', 'mu', [1, 1]*centi*poise, ...
                         'rho', [1000, 800], 'n', [1, 1], ...
                         'c', [1e-12; 1e-12], 'pRef', 0);
fm2.pcOW = @(sw) Pe + 0*sw;          % constant entry pressure, as the example

model2 = TwoPhaseOilWaterModelDP(G2, {rock_fracture2, rock_matrix2}, ...
                                     {ff2, fm2});
model2.transfer_model_object = KazemiTwoPhaseTransferFunction( ...
    'KazemiShapeFactor', [20, 20, 20]);
model2 = model2.validateModel();
% The two-phase case cannot reach 1e-12: the upstream switch in the transfer
% term makes the last few Newton steps chatter, and MRST then *cuts the
% timestep*, which would leave the two codes solving different time
% discretizations and is a far bigger error than the tolerance itself.
% 1e-10 is reachable in one shot, so both codes take the same steps.
model2.toleranceCNV = 1e-10;
model2.toleranceMB = 1e-12;
model2.nonlinearTolerance = 1e-10;

state2 = initResSol(G2, 0*psia, [0, 1]);        % both continua full of oil
state2.wellSol = initWellSolAD([], model2, state2);
state2.pressure_matrix = state2.pressure;
state2.sm = state2.s;

bc2 = pside([],  G2, 'xmin', 70*barsa, 'sat', [1, 0]);
bc2 = pside(bc2, G2, 'xmax',  0*barsa, 'sat', [1, 0]);

dt2 = 0.5*day;
n2  = 6;
p2  = zeros(G2.cells.num, n2);
sw2 = zeros(G2.cells.num, n2);
pm2 = zeros(G2.cells.num, n2);
swm2 = zeros(G2.cells.num, n2);
for k = 1:n2
    state2 = solver.solveTimestep(state2, dt2, model2, 'bc', bc2);
    p2(:, k)   = state2.pressure;
    sw2(:, k)  = state2.s(:, 1);
    pm2(:, k)  = state2.pressure_matrix;
    swm2(:, k) = state2.sm(:, 1);
end

ref.n2   = n2;
ref.dt2  = dt2;
ref.Pe   = Pe;
ref.p2   = p2;
ref.sw2  = sw2;
ref.pm2  = pm2;
ref.swm2 = swm2;

fprintf('2ph: at the inlet, fracture sW %.3f, matrix sW %.3f\n', ...
        sw2(1, end), swm2(1, end));

save(fullfile(here, 'ref_p15_dual_porosity.mat'), '-struct', 'ref', '-v7');
fprintf('Wrote Phase-15 dual-porosity reference\n');
