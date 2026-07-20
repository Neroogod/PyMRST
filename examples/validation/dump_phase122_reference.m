%% Dump Phase 122 reference: water compositional sequential transport step.
%
% Run in MATLAB after starting MRST. Writes ref_p122_comptransport_water.mat.
%
% This extends Phase 121 to MRST's water-active
% TransportOverallCompositionModel path. There are no wells or boundary
% conditions, so the reference isolates the water transport equation and the
% hydrocarbon component transport equations with frozen pressure fluxes.

mrstModule add ad-core ad-props compositional sequential

thisDir = fileparts(mfilename('fullpath'));
gravity reset off;

% -- grid, rock ---------------------------------------------------------------
nx = 8;
G = computeGeometry(cartGrid([nx, 1, 1], [80, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid and mixture --------------------------------------------------------
fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], ...
                           'rho', [1000, 700, 100], ...
                           'mu', [1, 1, 1]*centi*poise, ...
                           'c', [0, 0, 0]);

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

modelP = PressureOverallCompositionModel(G, rock, fluid, mixture, ...
                                         'water', true);
modelP.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
modelP.nonlinearTolerance = 1e-10;
try, modelP.toleranceCNV = 1e-10; end %#ok
try, modelP.toleranceMB  = 1e-12; end %#ok

modelT = TransportOverallCompositionModel(G, rock, fluid, mixture, ...
                                          'water', true);
modelT.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
modelT.nonlinearTolerance = 1e-10;
try, modelT.toleranceCNV = 1e-10; end %#ok
try, modelT.toleranceMB  = 1e-12; end %#ok

% -- previous state and pressure solve ---------------------------------------
T0 = 350;
p0 = linspace(78*barsa, 62*barsa, nx)';
p_guess = p0 + linspace(0.5*barsa, -0.35*barsa, nx)';
sW0 = linspace(0.18, 0.34, nx)';
z0 = [0.5, 0.3, 0.2];
s0 = [sW0, 0.5*(1 - sW0), 0.5*(1 - sW0)];

state0 = initCompositionalState(modelP, p0, T0, s0, z0, modelP.EOSModel);
state0.wellSol = initWellSolAD([], modelP, state0);
stateP = state0;
stateP.pressure = p_guess;

dt = 12*day;
forces = struct('W', [], 'bc', [], 'src', []);
solver = BackslashSolverAD();

maxIts = 40;
tol = 1e-10;
for itsP = 1:maxIts
    [problemP, stateP] = modelP.getEquations(state0, stateP, dt, forces, ...
                                             'iteration', itsP);
    resP = vertcat(value(problemP.equations{:}));
    if max(abs(resP)) < tol
        break
    end
    problemP = problemP.assembleSystem();
    [dxP, ~, ~] = solver.solveLinearProblem(problemP, modelP);
    stateP.pressure = stateP.pressure + dxP{1};
end
stateP = modelP.computeFlash(stateP, dt, 1);

% -- transport solve ----------------------------------------------------------
stateT = stateP;
for itsT = 1:maxIts
    [problemT, stateT] = modelT.getEquations(state0, stateT, dt, forces, ...
                                             'iteration', itsT);
    resT = vertcat(value(problemT.equations{:}));
    if max(abs(resT)) < tol
        break
    end
    problemT = problemT.assembleSystem();
    [dxT, ~, ~] = solver.solveLinearProblem(problemT, modelT);
    stateT.sT = stateT.sT + dxT{1};
    z = stateT.components;
    rm = zeros(size(z, 1), 1);
    for compNo = 1:(numel(names)-1)
        z0_comp = z(:, compNo);
        dz = dxT{compNo + 1};
        if isfinite(modelT.dzMaxAbs)
            dz = sign(dz).*min(abs(dz), modelT.dzMaxAbs);
        end
        z(:, compNo) = min(max(z0_comp + dz, 0), 1);
        rm = rm - (z(:, compNo) - z0_comp);
    end
    z(:, end) = min(max(z(:, end) + rm, 0), 1);
    z = bsxfun(@rdivide, z, sum(z, 2));
    stateT.components = z;
    stateT.s(:, 1) = min(max(stateT.s(:, 1) + dxT{numel(names) + 1}, 0), 1);
    stateT = modelT.computeFlash(stateT, dt, 1);
end
stateT_before_update = stateT;
stateT.s = bsxfun(@times, stateT.s, stateT.sT);

p_pressure = stateP.pressure;
z_pressure = stateP.components;
s_pressure = stateP.s;
flux_pressure = stateP.flux;
sT_transport = stateT_before_update.sT;
z_transport = stateT.components;
s_transport = stateT.s;
p_transport = stateT.pressure;

save(fullfile(thisDir, 'ref_p122_comptransport_water.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'p_guess', 'sW0', 'z0', 'dt', ...
     'p_pressure', 'z_pressure', 's_pressure', 'flux_pressure', ...
     'sT_transport', 'z_transport', 's_transport', 'p_transport', ...
     'itsP', 'itsT', 'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p122_comptransport_water.mat to %s\n', thisDir);
fprintf('  %d cells, pressure %d iterations, transport %d iterations\n', ...
        nx, itsP, itsT);
