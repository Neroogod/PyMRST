%% Dump Phase 44 reference: adaptive timestep selection, from MRST ad-core.
%
% Run in MATLAB after starting MRST. Writes ref_p44_timesteps.mat.
%
% A timestep selector is a *controller*, and the honest way to test a
% controller is to drive it with a known input and record what it does - not to
% run a simulation and hope the physics agrees, which tests the physics as much
% as the controller and tells you nothing when it disagrees.
%
% So each selector is fed a fixed, synthetic history - "this step took 3
% Newton iterations, this one took 12" - and every timestep it picks is
% recorded. That pins the control law exactly, independently of the model.
%
% Four cases:
%   A. IterationCountTimeStepSelector, all steps converged. Exercises the
%      two-point (`restart == false`) branch, which is the one that actually
%      does the work and the one easiest to get wrong.
%   B. The same, with failures salted in - which forces the one-point
%      `restart` branch, because a controller must not learn from a step that
%      the physics, rather than the controller, chose.
%   C. StateChangeTimeStepSelector, targeting a relative change in pressure.
%   D. rampupTimesteps.

mrstModule add ad-core ad-props ad-blackoil

gravity reset off;
thisDir = fileparts(mfilename('fullpath'));

% A model and a solver, only so the selectors have something to ask about.
% (They use model.stepFunctionIsLinear and solver.maxIterations, nothing more.)
G    = computeGeometry(cartGrid([3, 3, 1], [30, 30, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 5]*centi*poise, ...
                           'rho', [1000, 800], 'n', [2, 2], 'pRef', 200*barsa);
model  = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                              'gas', false);
model  = model.validateModel();
solver = NonLinearSolver('maxIterations', 12);

forces = model.getValidDrivingForces();
forces.controlId = 1;

its_A = [3, 8, 5, 2, 12, 6, 4, 4, 9, 5, 5, 5, 7, 3, 2, 6];
conv_B = logical([1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);

% -- A: iteration count, everything converges ---------------------------------
sel = IterationCountTimeStepSelector('targetIterationCount', 5, ...
                                     'iterationOffset', 5, ...
                                     'maxTimestep', 100*day, ...
                                     'minTimestep', 0.01*day);
sel.newControlStep(forces);
dt = 10*day;  dt_prev = nan;
A_dt = zeros(numel(its_A), 1);
for k = 1:numel(its_A)
    dt = sel.pickTimestep(dt_prev, dt, model, solver, [], [], forces);
    A_dt(k) = dt;
    rep = struct('Iterations', its_A(k), 'Timestep', dt, 'Converged', true);
    sel.storeTimestep(rep);
    dt_prev = dt;
end

% -- B: iteration count, with failed steps ------------------------------------
sel = IterationCountTimeStepSelector('targetIterationCount', 5, ...
                                     'iterationOffset', 5, ...
                                     'maxTimestep', 100*day, ...
                                     'minTimestep', 0.01*day);
sel.newControlStep(forces);
dt = 10*day;  dt_prev = nan;
B_dt = zeros(numel(its_A), 1);
for k = 1:numel(its_A)
    if k > 1 && ~conv_B(k-1)
        dt = sel.cutTimestep(dt_prev, dt, model, solver, [], [], forces);
    else
        dt = sel.pickTimestep(dt_prev, dt, model, solver, [], [], forces);
    end
    B_dt(k) = dt;
    rep = struct('Iterations', its_A(k), 'Timestep', dt, ...
                 'Converged', conv_B(k));
    sel.storeTimestep(rep);
    if conv_B(k)
        dt_prev = dt;
    end
end

% -- C: state change, targeting a relative pressure change --------------------
% A synthetic pair of states per step: the pressure moves by a prescribed
% relative amount, and the selector must bound it.
moves = [0.02, 0.10, 0.01, 0.20, 0.05, 0.05, 0.30, 0.005];
sel = StateChangeTimeStepSelector('targetProps', {'pressure'}, ...
                                  'targetChangeRel', 0.05, ...
                                  'relaxFactor', 1, ...
                                  'targetIterationCount', 5, ...
                                  'iterationOffset', 5, ...
                                  'maxTimestep', 100*day, ...
                                  'minTimestep', 0.01*day);
sel.newControlStep(forces);
dt = 10*day;  dt_prev = nan;
C_dt = zeros(numel(moves), 1);
p0 = repmat(200*barsa, G.cells.num, 1);
for k = 1:numel(moves)
    st_prev = struct('pressure', p0);
    st_curr = struct('pressure', p0 * (1 + moves(k)));
    dt = sel.pickTimestep(dt_prev, dt, model, solver, st_prev, st_curr, forces);
    C_dt(k) = dt;
    rep = struct('Iterations', 5, 'Timestep', dt, 'Converged', true);
    sel.storeTimestep(rep);
    dt_prev = dt;
end

% -- D: rampupTimesteps -------------------------------------------------------
D_year   = rampupTimesteps(365*day, 30*day, 8);
D_short  = rampupTimesteps(100*day, 30*day, 5);
D_tiny   = rampupTimesteps(10*day,  30*day, 8);

save(fullfile(thisDir, 'ref_p44_timesteps.mat'), '-v7', ...
     'its_A', 'conv_B', 'moves', 'A_dt', 'B_dt', 'C_dt', ...
     'D_year', 'D_short', 'D_tiny');

fprintf('Wrote ref_p44_timesteps.mat to %s\n', thisDir);
fprintf('  A (iteration count):  %s\n', mat2str(round(A_dt'/day, 3)));
fprintf('  B (with failures):    %s\n', mat2str(round(B_dt'/day, 3)));
fprintf('  C (state change):     %s\n', mat2str(round(C_dt'/day, 3)));
fprintf('  D (rampup, 1 year):   %d steps\n', numel(D_year));
