%% Dump Phase 56 reference: overall-composition flow with water.
%
% Run in MATLAB after starting MRST. Writes ref_p56_compwater.mat.
%
% This checks the first water slice for `equationsCompositional`: water on,
% no wells, no gravity, no capillary pressure. A closed 1D domain starts from a
% pressure gradient, nonuniform water saturation, and uniform hydrocarbon
% composition. Total water and hydrocarbon component masses must be conserved.

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));
gravity reset off;

% -- grid, rock ---------------------------------------------------------------
nx = 10;
G = computeGeometry(cartGrid([nx, 1, 1], [100, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid: water/oil/gas, Corey n=2, incompressible water --------------------
fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], ...
                           'rho', [1000, 700, 100], ...
                           'mu', [1, 1, 1]*centi*poise, ...
                           'c', [0, 0, 0]);

% -- the same mixture PyMRST's simple_mixture() builds ------------------------
names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

model = OverallCompositionCompositionalModel(G, rock, fluid, mixture, ...
                                             'water', true);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-10;
try, model.toleranceCNV = 1e-10; end %#ok
try, model.toleranceMB  = 1e-12; end %#ok

% -- initial state: pressure gradient, water gradient, uniform z --------------
T0 = 350;
p0 = linspace(80*barsa, 60*barsa, nx)';
sW0 = linspace(0.15, 0.35, nx)';
z0 = [0.5, 0.3, 0.2];
s0 = [sW0, 0.5*(1 - sW0), 0.5*(1 - sW0)];
state0 = initCompositionalState(model, p0, T0, s0, z0, model.EOSModel);

% -- schedule: closed domain, no wells, 5 x 20 days ---------------------------
nstep = 5;
dt = 20*day;
schedule = simpleSchedule(repmat(dt, nstep, 1));

nls = NonLinearSolver('maxIterations', 60);
[~, states] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls);

ncomp = numel(names);
p_steps  = zeros(nx, nstep);
sw_steps = zeros(nx, nstep);
z_steps  = zeros(nx, ncomp, nstep);
sO_steps = zeros(nx, nstep);
sG_steps = zeros(nx, nstep);
for k = 1:nstep
    p_steps(:, k)    = states{k}.pressure;
    sw_steps(:, k)   = states{k}.s(:, 1);
    z_steps(:, :, k) = states{k}.components;
    sO_steps(:, k)   = states{k}.s(:, 2);
    sG_steps(:, k)   = states{k}.s(:, 3);
end

save(fullfile(thisDir, 'ref_p56_compwater.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'sW0', 'z0', 'dt', 'nstep', 'ncomp', ...
     'p_steps', 'sw_steps', 'z_steps', 'sO_steps', 'sG_steps', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p56_compwater.mat to %s\n', thisDir);
fprintf('  %d cells, %d steps of %.0f days, water on\n', nx, nstep, dt/day);
fprintf('  final sW: %.4f - %.4f, pressure spread %.4f bar\n', ...
        min(sw_steps(:,end)), max(sw_steps(:,end)), ...
        (max(p_steps(:,end))-min(p_steps(:,end)))/barsa);
