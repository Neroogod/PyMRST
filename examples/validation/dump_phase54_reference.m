%% Dump Phase 54 reference: compositional flow with component-aware wells.
%
% Run in MATLAB after starting MRST. Writes ref_p54_compwells.mat.
%
% End-to-end check of `equationsCompositional` with wells: a 1D
% injector/producer pair where the injector streams a methane-rich gas
% (`W.components`) at a fixed surface rate and the producer is BHP-controlled.
% Component mass sources follow `getExtraWellContributions` in
% `ThreePhaseCompositionalModel`.
%
% Nonlinear tolerances are tightened below MRST's 1e-3 default so the
% reference is a converged answer, not MRST's slack.

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));

% -- grid, rock ---------------------------------------------------------------
nx = 10;
G = computeGeometry(cartGrid([nx, 1, 1], [100, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid: two-phase oil/gas, Corey n=2 --------------------------------------
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
                           'rho', [700, 100], 'mu', [1, 1]*centi*poise, ...
                           'c', [0, 0]);
fluid.rhoSinj = [700, 100];

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
model.FacilityModel.toleranceWellRate = 1e-12;
model.FacilityModel.toleranceWellBHP  = 1e-6;

% -- initial composition (also used as producer injection stream if needed)
z0 = [0.5, 0.3, 0.2];

% -- wells: gas injector at left, BHP producer at right -----------------------
injRate = 5e-5;
prodBhp = 55*barsa;
W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', injRate, ...
            'Comp_i', [0, 1], 'Radius', 0.1, 'Sign', 1, ...
            'components', [0.9, 0.05, 0.05]);
W = addWell(W, G, rock, nx, 'Type', 'bhp', 'Val', prodBhp, ...
            'Comp_i', [0, 1], 'Radius', 0.1, 'Sign', -1, ...
            'components', z0);

% -- initial state: uniform pressure and composition, isothermal --------------
T0 = 350;
p0 = 70*barsa;
state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);

% -- schedule: 5 x 30 days ----------------------------------------------------
nstep = 5;
dt = 30*day;
schedule = simpleSchedule(repmat(dt, nstep, 1), 'W', W);

nls = NonLinearSolver('maxIterations', 50);
[wellSols, states] = simulateScheduleAD(state0, model, schedule, ...
                                        'NonLinearSolver', nls);

ncomp = numel(names);
p_steps  = zeros(nx, nstep);
z_steps  = zeros(nx, ncomp, nstep);
sO_steps = zeros(nx, nstep);
qOs = zeros(nstep, 2);
qGs = zeros(nstep, 2);
bhp = zeros(nstep, 2);
for k = 1:nstep
    p_steps(:, k)    = states{k}.pressure;
    z_steps(:, :, k) = states{k}.components;
    sO_steps(:, k)   = states{k}.s(:, 1);
    qOs(k, :) = [wellSols{k}.qOs];
    qGs(k, :) = [wellSols{k}.qGs];
    bhp(k, :) = [wellSols{k}.bhp];
end

WI_inj = W(1).WI;
WI_prod = W(2).WI;

save(fullfile(thisDir, 'ref_p54_compwells.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'z0', 'dt', 'nstep', 'ncomp', ...
     'injRate', 'prodBhp', 'p_steps', 'z_steps', 'sO_steps', ...
     'qOs', 'qGs', 'bhp', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw', 'WI_inj', 'WI_prod');

fprintf('Wrote ref_p54_compwells.mat to %s\n', thisDir);
fprintf('  %d cells, %d steps of %.0f days, inj rate %.2e, prod bhp %.1f bar\n', ...
        nx, nstep, dt/day, injRate, prodBhp/barsa);
fprintf('  final z at injector: [%.4f %.4f %.4f]\n', z_steps(1, :, end));
