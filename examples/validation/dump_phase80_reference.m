%% Dump Phase 80 reference: natural-variable compositional model.
%
% Run in MATLAB after starting MRST. Writes ref_p80_natvars.mat.

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));

nx = 4;
G = computeGeometry(cartGrid([nx, 1, 1], [40, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
                           'rho', [700, 100], 'mu', [1, 1]*centi*poise, ...
                           'c', [0, 0]);
fluid.rhoSinj = [700, 100];

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

model = NaturalVariablesCompositionalModel(G, rock, fluid, mixture, ...
                                           'water', false);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-10;
try, model.toleranceCNV = 1e-10; end %#ok
try, model.toleranceMB  = 1e-12; end %#ok
model.FacilityModel.toleranceWellRate = 1e-12;
model.FacilityModel.toleranceWellBHP  = 1e-6;

z0 = [0.5, 0.3, 0.2];
T0 = 350;
p0 = 50*barsa;
state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);

injRate = 1e-5;
prodBhp = 45*barsa;
W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', injRate, ...
            'Comp_i', [0, 1], 'Radius', 0.1, 'Sign', 1, ...
            'components', [0.9, 0.05, 0.05]);
W = addWell(W, G, rock, nx, 'Type', 'bhp', 'Val', prodBhp, ...
            'Comp_i', [0, 1], 'Radius', 0.1, 'Sign', -1, ...
            'components', z0);

dt = 0.5*day;
schedule = simpleSchedule(dt, 'W', W);
nls = NonLinearSolver('maxIterations', 50);
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule, ...
                                                'NonLinearSolver', nls);
state = states{1};
wellSol = wellSols{1};

p = state.pressure;
z = state.components;
x = state.x;
y = state.y;
sO = state.s(:, 1);
sG = state.s(:, 2);
qOs = [wellSol.qOs];
qGs = [wellSol.qGs];
bhp = [wellSol.bhp];
iterations = report.Iterations;
twoPhase = state.flag == 0;

save(fullfile(thisDir, 'ref_p80_natvars.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'z0', 'dt', 'injRate', 'prodBhp', ...
     'p', 'z', 'x', 'y', 'sO', 'sG', 'qOs', 'qGs', 'bhp', ...
     'iterations', 'twoPhase', 'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p80_natvars.mat to %s\n', thisDir);
fprintf('  natural variables: %d cells, dt %.2f days, iterations %d\n', ...
        nx, dt/day, iterations);
