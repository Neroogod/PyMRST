%% Dump Phase 60 reference: compositional pressure-only Newton step.
%
% Run in MATLAB after starting MRST. Writes ref_p60_comppressurestep.mat.
%
% This validates the pressure solve that uses `opt.pressure`: MRST's
% `PressureOverallCompositionModel` assembles the weighted pressure equation
% with full composition derivatives, then solves only pressure.

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));
gravity reset off;

% -- grid, rock ---------------------------------------------------------------
nx = 8;
G = computeGeometry(cartGrid([nx, 1, 1], [80, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid and mixture --------------------------------------------------------
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
                           'rho', [700, 100], ...
                           'mu', [1, 1]*centi*poise, ...
                           'c', [0, 0]);

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

model = PressureOverallCompositionModel(G, rock, fluid, mixture, ...
                                        'water', false);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-10;
try, model.toleranceCNV = 1e-10; end %#ok
try, model.toleranceMB  = 1e-12; end %#ok

% -- previous state and initial pressure guess -------------------------------
T0 = 350;
p0 = linspace(78*barsa, 62*barsa, nx)';
p_guess = p0 + linspace(0.6*barsa, -0.4*barsa, nx)';
z0 = [0.5, 0.3, 0.2];

state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);
state0.wellSol = initWellSolAD([], model, state0);
stateGuess = state0;
stateGuess.pressure = p_guess;

dt = 12*day;
forces = struct('W', [], 'bc', [], 'src', []);

stateP = stateGuess;
maxIts = 40;
tol = 1e-10;
for its = 1:maxIts
    [problem, stateP] = model.getEquations(state0, stateP, dt, forces, ...
                                           'iteration', its);
    res = value(problem.equations{1});
    if max(abs(res)) < tol
        break
    end
    problem = problem.assembleSystem();
    dx = problem.A \ problem.b;
    stateP.pressure = stateP.pressure + dx;
end

p_pressure = stateP.pressure;
z_pressure = stateP.components;

save(fullfile(thisDir, 'ref_p60_comppressurestep.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'p_guess', 'z0', 'dt', 'p_pressure', 'z_pressure', ...
     'its', 'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p60_comppressurestep.mat to %s\n', thisDir);
fprintf('  %d cells, pressure solve in %d iterations, pressure spread %.4f bar\n', ...
        nx, its, (max(p_pressure)-min(p_pressure))/barsa);
