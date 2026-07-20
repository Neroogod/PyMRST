function dump_phase81_reference()
% Phase 81 reference: water/oil/gas natural-variable compositional model.

mrstModule add ad-core ad-props compositional

nx = 4;
G = cartGrid([nx, 1, 1], [40, 10, 10]);
G = computeGeometry(G);
rock = makeRock(G, 100*milli*darcy, 0.25);

names = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit = [190.564, 425.125, 617.7];
Pcrit = [4599200.0, 3796000.0, 2103000.0];
Vcrit = [9.86e-5, 2.55e-4, 6.098e-4];
acf = [0.0114, 0.201, 0.4884];
mw = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

fluid = initSimpleADIFluid('phases', 'WOG', ...
    'n', [2, 2, 2], ...
    'rho', [1000, 700, 100], ...
    'mu', [1, 1, 1]*centi*poise, ...
    'c', [0, 0, 0]);
fluid.rhoSinj = [1000, 700, 100];

model = NaturalVariablesCompositionalModel(G, rock, fluid, mixture, 'water', true);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-8;
model.verbose = false;
model.FacilityModel.toleranceWellRate = 1e-12;
model.FacilityModel.toleranceWellBHP = 1e-8;

z0 = [0.5, 0.3, 0.2];
T0 = 350;
p0 = 50*barsa;
sw0 = 0.2;
state0 = initCompositionalState(model, p0, T0, [sw0, 0.4, 0.4], z0, model.EOSModel);

W = [];
W = verticalWell(W, G, rock, 1, 1, 1, 'type', 'rate', 'val', 1e-5, ...
    'comp_i', [1, 0, 0], 'sign', 1);
W(1).components = [0.5, 0.3, 0.2];
W = verticalWell(W, G, rock, nx, 1, 1, 'type', 'bhp', 'val', 45*barsa, ...
    'comp_i', [0, 0.5, 0.5], 'sign', -1);
W(2).components = z0;

dt = 0.25*day;
schedule = simpleSchedule(dt, 'W', W);
solver = NonLinearSolver('maxIterations', 25, 'errorOnFailure', true, 'verbose', false);
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', solver);

state = states{1};
ws = wellSols{1};
qWs = vertcat(ws.qWs);
qOs = vertcat(ws.qOs);
qGs = vertcat(ws.qGs);
bhp = vertcat(ws.bhp);
cr = report.ControlstepReports{1};
if isfield(cr, 'NonlinearReport')
    nr = cr.NonlinearReport{1};
    if isfield(nr, 'Iterations')
        iterations = nr.Iterations;
    else
        iterations = -1;
    end
elseif isfield(cr, 'StepReports') && isfield(cr.StepReports{1}, 'NonlinearReport')
    nr = cr.StepReports{1}.NonlinearReport{1};
    if isfield(nr, 'Iterations')
        iterations = nr.Iterations;
    else
        iterations = -1;
    end
elseif isfield(cr, 'Iterations')
    iterations = cr.Iterations;
else
    iterations = -1;
end

p = state.pressure;
z = state.components;
x = state.x;
y = state.y;
sW = state.s(:, 1);
sO = state.s(:, 2);
sG = state.s(:, 3);

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p81_natvars_water.mat');
save(out, 'nx', 'T0', 'p0', 'z0', 'sw0', 'dt', 'p', 'z', 'x', 'y', ...
    'sW', 'sO', 'sG', 'qWs', 'qOs', 'qGs', 'bhp', 'iterations', ...
    'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');
fprintf('Wrote ref_p81_natvars_water.mat to %s\n', fileparts(out));
fprintf('  water natural variables: %d cells, dt %.2f days, iterations %d\n', ...
    nx, dt/day, iterations);
end
