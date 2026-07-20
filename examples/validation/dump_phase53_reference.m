%% Dump Phase 53 reference: overall-composition flow, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p53_compflow.mat.
%
% This is the end-to-end check of the compositional *flow* model
% (`equationsCompositional`): the accumulation and flux, solved fully
% implicitly, with every underlying piece (flash + derivatives, densities,
% viscosities, mass fractions, saturations) already validated on its own in
% Phases 13/50/51/52.
%
% To exercise the residual without dragging in the well model (that is the next
% phase), the drive is a **non-uniform initial state in a closed domain**: a 1D
% column started with a pressure gradient and left to equilibrate under no-flow
% boundaries and no wells. The fluid redistributes - accumulation and flux both
% active - and, with no sources, the total mass of every component must be
% conserved, a check the reference cannot fake.
%
% The nonlinear tolerances are tightened well below MRST's 1e-3 default: at the
% default, MRST stops before its own equations are solved, and a comparison
% claiming better than ~1e-5 against it would be measuring MRST's slack rather
% than agreement (see the Norne notes in the README).

mrstModule add ad-core ad-props compositional

thisDir = fileparts(mfilename('fullpath'));

% -- grid, rock ---------------------------------------------------------------
nx = 10;
G = computeGeometry(cartGrid([nx, 1, 1], [100, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid: two-phase oil/gas, Corey n=2; densities/viscosity here are unused
%    by the residual (compositional takes rho, mu from the EOS/LBC), but
%    initSimpleADIFluid requires them. Incompressible rock (no pv multiplier).
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
% Tighten convergence so we compare converged answers, not MRST's slack.
model.nonlinearTolerance = 1e-10;
try, model.toleranceCNV = 1e-10; end %#ok
try, model.toleranceMB  = 1e-12; end %#ok

% -- initial state: a pressure gradient, uniform composition, isothermal ------
T0 = 350;
p0 = linspace(80*barsa, 60*barsa, nx)';
z0 = [0.5, 0.3, 0.2];
state0 = initCompositionalState(model, p0, T0, [0.5, 0.5], z0, model.EOSModel);

% -- schedule: closed domain, no wells, 5 x 20 days ---------------------------
nstep = 5;
dt = 20*day;
schedule = simpleSchedule(repmat(dt, nstep, 1));   % empty W by default

nls = NonLinearSolver('maxIterations', 50);
[~, states] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls);

ncomp = numel(names);
p_steps  = zeros(nx, nstep);
z_steps  = zeros(nx, ncomp, nstep);
sO_steps = zeros(nx, nstep);
sG_steps = zeros(nx, nstep);
for k = 1:nstep
    p_steps(:, k)    = states{k}.pressure;
    z_steps(:, :, k) = states{k}.components;
    sO_steps(:, k)   = states{k}.s(:, 1);
    sG_steps(:, k)   = states{k}.s(:, 2);
end

save(fullfile(thisDir, 'ref_p53_compflow.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'z0', 'dt', 'nstep', 'ncomp', ...
     'p_steps', 'z_steps', 'sO_steps', 'sG_steps', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p53_compflow.mat to %s\n', thisDir);
fprintf('  %d cells, %d steps of %.0f days\n', nx, nstep, dt/day);
fprintf('  p spread: initial %.2f bar -> final %.2f bar\n', ...
        (max(p0)-min(p0))/barsa, ...
        (max(p_steps(:,end))-min(p_steps(:,end)))/barsa);
fprintf('  sO final: %.4f - %.4f\n', min(sO_steps(:,end)), max(sO_steps(:,end)));
