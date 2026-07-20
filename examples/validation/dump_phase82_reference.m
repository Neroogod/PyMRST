function dump_phase82_reference()
% Phase 82 reference: overall-composition compositional boundary conditions.

mrstModule add ad-core ad-props compositional

nx = 8;
ny = 4;
G = cartGrid([nx, ny, 1], [1000, 1000, 10]);
G = computeGeometry(G);
rock = makeRock(G, 100*milli*darcy, 0.2);

names = {'Methane', 'CarbonDioxide', 'n-Decane'};
Tcrit = [190.5640, 304.1282, 617.7000];
Pcrit = [4599200, 7377300, 2103000];
Vcrit = [9.8628e-05, 9.4118e-05, 6.0976e-04];
acf = [0.0114, 0.2239, 0.4884];
mw = [0.0160, 0.0440, 0.1423];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

flowfluid = initSimpleADIFluid('rho', [1000, 500, 500], ...
    'mu', [1, 1, 1]*centi*poise, ...
    'n', [2, 2, 2], ...
    'c', [1e-5, 0, 0]/barsa);

minP = 75*barsa;
pv = poreVolume(G, rock);
totTimeForRate = 20*year;
bc = [];
bc = fluxside(bc, G, 'xmin', sum(pv)/totTimeForRate, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', minP, 'sat', [1, 0]);
bc.components = repmat([0.1, 0.9, 0], numel(bc.face), 1);

T0 = 423.15;
z0 = [0.3, 0.1, 0.6];
p0 = 1.5*minP;
state0 = initResSol(G, p0, [1, 0]);
state0.T = repmat(T0, G.cells.num, 1);
state0.components = repmat(z0, G.cells.num, 1);

dt = [0.125; 0.125; 0.25]*day;
schedule = simpleSchedule(dt, 'bc', bc);
model = OverallCompositionCompositionalModel(G, rock, flowfluid, mixture, 'water', false);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-8;
model.verbose = false;
solver = NonLinearSolver('maxIterations', 50, 'errorOnFailure', true, ...
    'useRelaxation', true, 'verbose', false);
[~, states, report] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', solver);

nstep = numel(states);
nc = G.cells.num;
p = zeros(nc, nstep);
z = zeros(nc, mixture.getNumberOfComponents(), nstep);
sO = zeros(nc, nstep);
sG = zeros(nc, nstep);
for i = 1:nstep
    p(:, i) = states{i}.pressure;
    z(:, :, i) = states{i}.components;
    sO(:, i) = states{i}.s(:, 1);
    sG(:, i) = states{i}.s(:, 2);
end
iterations = report.Iterations;
bcFaces = bc.face;
bcValues = bc.value;
bcComponents = bc.components;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p82_compositional_bc.mat');
save(out, 'nx', 'ny', 'T0', 'p0', 'z0', 'dt', 'p', 'z', 'sO', 'sG', ...
    'iterations', 'bcFaces', 'bcValues', 'bcComponents', ...
    'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');
fprintf('Wrote ref_p82_compositional_bc.mat to %s\n', fileparts(out));
fprintf('  compositional BC schedule: %d cells, %d steps, total %.2f days\n', ...
    nc, nstep, sum(dt)/day);
end
