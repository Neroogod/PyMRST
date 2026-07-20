%% Dump Phase 55 reference: overall-composition flow with gravity.
%
% Run in MATLAB after starting MRST. Writes ref_p55_compgravity.mat.
%
% This checks the gravity part of `equationsCompositional` without wells or
% water: a closed vertical oil/gas column starts from uniform pressure and
% composition, then segregates under gravity. With no sources, total component
% mass must remain conserved.

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));
gravity reset on;

% -- grid, rock ---------------------------------------------------------------
nz = 10;
G = computeGeometry(cartGrid([1, 1, nz], [10, 10, 100]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid: two-phase oil/gas, Corey n=2 --------------------------------------
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
                           'rho', [700, 100], 'mu', [1, 1]*centi*poise, ...
                           'c', [0, 0]);

% -- the same mixture PyMRST's simple_mixture() builds ------------------------
names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

model = OverallCompositionCompositionalModel(G, rock, fluid, mixture, ...
                                             'water', false);
model.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
model.nonlinearTolerance = 1e-10;
try, model.toleranceCNV = 1e-10; end %#ok
try, model.toleranceMB  = 1e-12; end %#ok

% -- initial state: uniform pressure/composition, isothermal ------------------
T0 = 350;
p0 = 70*barsa;
z0 = [0.5, 0.3, 0.2];
state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);

% -- schedule: closed domain, no wells, 5 x 20 days ---------------------------
nstep = 5;
dt = 20*day;
schedule = simpleSchedule(repmat(dt, nstep, 1));

nls = NonLinearSolver('maxIterations', 60);
[~, states] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls);

ncomp = numel(names);
p_steps  = zeros(nz, nstep);
z_steps  = zeros(nz, ncomp, nstep);
sO_steps = zeros(nz, nstep);
sG_steps = zeros(nz, nstep);
for k = 1:nstep
    p_steps(:, k)    = states{k}.pressure;
    z_steps(:, :, k) = states{k}.components;
    sO_steps(:, k)   = states{k}.s(:, 1);
    sG_steps(:, k)   = states{k}.s(:, 2);
end

z_centroids = G.cells.centroids(:, 3);
g_acc = norm(gravity);

save(fullfile(thisDir, 'ref_p55_compgravity.mat'), '-v7', ...
     'nz', 'T0', 'p0', 'z0', 'dt', 'nstep', 'ncomp', 'g_acc', ...
     'z_centroids', 'p_steps', 'z_steps', 'sO_steps', 'sG_steps', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p55_compgravity.mat to %s\n', thisDir);
fprintf('  %d vertical cells, %d steps of %.0f days, gravity %.5f m/s2\n', ...
        nz, nstep, dt/day, g_acc);
fprintf('  final pressure spread %.4f bar, z top/bottom methane %.4f / %.4f\n', ...
        (max(p_steps(:,end))-min(p_steps(:,end)))/barsa, ...
        z_steps(1, 1, end), z_steps(end, 1, end));

gravity reset off;
