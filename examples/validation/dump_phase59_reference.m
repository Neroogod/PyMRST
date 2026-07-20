%% Dump Phase 59 reference: compositional opt.pressure pressure equation.
%
% Run in MATLAB after starting MRST. Writes ref_p59_comppressure.mat.
%
% This validates the equation-assembly slice used by MRST's
% `PressureOverallCompositionModel`: `equationsCompositional(...,
% 'pressure', true)` collapses the component conservation equations into one
% weighted pressure equation using `getPartialVolumes`.

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

% -- previous and current states ---------------------------------------------
T0 = 350;
p0 = linspace(78*barsa, 62*barsa, nx)';
p1 = p0 + linspace(0.6*barsa, -0.4*barsa, nx)';
z0 = [0.5, 0.3, 0.2];
z1 = [repmat(z0, nx, 1)];
z1(:, 1) = z1(:, 1) + linspace(0.01, -0.01, nx)';
z1(:, 2) = z1(:, 2) + linspace(-0.005, 0.005, nx)';
z1(:, 3) = 1 - z1(:, 1) - z1(:, 2);

state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);
state  = initCompositionalState(model, p1, T0, [0.5, 0.5], z1, model.EOSModel);
state0.wellSol = initWellSolAD([], model, state0);
state.wellSol  = initWellSolAD([], model, state);

dt = 12*day;
forces = struct('W', [], 'bc', [], 'src', []);

[problem, stateP] = model.getEquations(state0, state, dt, forces, ...
                                       'iteration', 1);
peq = value(problem.equations{1});
jacP = full(problem.equations{1}.jac{1});
weights = stateP.w;

save(fullfile(thisDir, 'ref_p59_comppressure.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'p1', 'z0', 'z1', 'dt', 'peq', 'jacP', 'weights', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p59_comppressure.mat to %s\n', thisDir);
fprintf('  %d cells, pressure residual max %.3e, weight range %.3e %.3e\n', ...
        nx, max(abs(peq)), min(weights(:)), max(weights(:)));
