%% Dump Phase 57 reference: compositional water wells.
%
% Run in MATLAB after starting MRST. Writes ref_p57_compwaterwells.mat.
%
% This checks `model.water = true` with wells: a water injector at fixed
% surface rate and a BHP producer. Gravity and capillary pressure are off.

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
fluid.rhoSinj = [1000, 700, 100];

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
model.FacilityModel.toleranceWellRate = 1e-12;
model.FacilityModel.toleranceWellBHP  = 1e-6;

% -- wells: water injector at left, BHP producer at right ---------------------
z0 = [0.5, 0.3, 0.2];
injRate = 5e-5;
prodBhp = 55*barsa;
W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', injRate, ...
            'Comp_i', [1, 0, 0], 'Radius', 0.1, 'Sign', 1, ...
            'components', z0);
W = addWell(W, G, rock, nx, 'Type', 'bhp', 'Val', prodBhp, ...
            'Comp_i', [0.2, 0.4, 0.4], 'Radius', 0.1, 'Sign', -1, ...
            'components', z0);

% -- initial state ------------------------------------------------------------
T0 = 350;
p0 = 70*barsa;
sW0 = 0.2;
s0 = [sW0, 0.4, 0.4];
state0 = initCompositionalState(model, p0, T0, s0, z0, model.EOSModel);

% -- schedule: 5 x 30 days ----------------------------------------------------
nstep = 5;
dt = 30*day;
schedule = simpleSchedule(repmat(dt, nstep, 1), 'W', W);

nls = NonLinearSolver('maxIterations', 60);
[wellSols, states] = simulateScheduleAD(state0, model, schedule, ...
                                        'NonLinearSolver', nls);

ncomp = numel(names);
p_steps  = zeros(nx, nstep);
sw_steps = zeros(nx, nstep);
z_steps  = zeros(nx, ncomp, nstep);
sO_steps = zeros(nx, nstep);
sG_steps = zeros(nx, nstep);
qWs = zeros(nstep, 2);
qOs = zeros(nstep, 2);
qGs = zeros(nstep, 2);
bhp = zeros(nstep, 2);
for k = 1:nstep
    p_steps(:, k)    = states{k}.pressure;
    sw_steps(:, k)   = states{k}.s(:, 1);
    z_steps(:, :, k) = states{k}.components;
    sO_steps(:, k)   = states{k}.s(:, 2);
    sG_steps(:, k)   = states{k}.s(:, 3);
    qWs(k, :) = [wellSols{k}.qWs];
    qOs(k, :) = [wellSols{k}.qOs];
    qGs(k, :) = [wellSols{k}.qGs];
    bhp(k, :) = [wellSols{k}.bhp];
end

WI_inj = W(1).WI;
WI_prod = W(2).WI;

save(fullfile(thisDir, 'ref_p57_compwaterwells.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'sW0', 'z0', 'dt', 'nstep', 'ncomp', ...
     'injRate', 'prodBhp', 'p_steps', 'sw_steps', 'z_steps', ...
     'sO_steps', 'sG_steps', 'qWs', 'qOs', 'qGs', 'bhp', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw', 'WI_inj', 'WI_prod');

fprintf('Wrote ref_p57_compwaterwells.mat to %s\n', thisDir);
fprintf('  %d cells, %d steps of %.0f days, water inj %.2e, prod bhp %.1f bar\n', ...
        nx, nstep, dt/day, injRate, prodBhp/barsa);
fprintf('  final injector sW %.4f, qWs injector %.2e\n', ...
        sw_steps(1,end), qWs(end, 1));
