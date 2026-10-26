% blackoilTutorialMexAcceleration in MRST 2024a on a 20x20x10 grid (the
% script's physical size), for validate_ad_oo_blackoil_tutorial_mex_acceleration.py.
% MRST's MEX does not build on this machine, so the diagonal backend runs
% with useMex false and the run uses BackslashSolverAD; and AMGCL_CPRSolverAD's
% preparation of the first Newton system - everything it does in MATLAB
% before calling AMGCL - is recorded through CaptureAMGCL_CPRSolverAD.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat ad-props linearsolvers
addpath(fullfile(here, 'matlab_helpers'));
dims = [20, 20, 10];
pdims = [1000, 1000, 100]*meter;
G = cartGrid(dims, pdims);
G = computeGeometry(G);
rock = makeRock(G, 0.1*darcy, 0.3);
gravity reset on
time = 10*year;
irate = 0.3*sum(poreVolume(G, rock)/time);
W = [];
W = verticalWell(W, G, rock, 1, 1, [], 'comp_i', [1, 0, 0], ...
                'type', 'rate', 'val', irate);
W = verticalWell(W, G, rock, dims(1), dims(2), 1, 'comp_i', [1, 0, 0], ...
                 'type', 'bhp', 'val', 100*barsa);
fluid = initSimpleADIFluid('n', [2, 3, 1], ...
                           'rho', [1000, 700, 10], ...
                           'mu', [1, 5, 0.1]*centi*poise, ...
                           'c', [0, 1e-6, 1e-4]/barsa);
dt = repmat(30*day, 5, 1);
schedule = simpleSchedule(dt, 'W', W);
model = GenericBlackOilModel(G, rock, fluid, 'disgas', false, 'vapoil', false);
model.AutoDiffBackend = DiagonalAutoDiffBackend('useMex', false);
model = model.validateModel();
model.FacilityModel.primaryVariableSet = 'bhp';
ncomp = model.getNumberOfComponents();
region = getInitializationRegionsBlackOil(model, [70, 30], 'datum_pressure', 200*barsa);
state0 = initStateBlackOilAD(model, region);
% The run, with a direct linear solver
nls = NonLinearSolver('LinearSolver', BackslashSolverAD());
[ws, states, rep] = simulateScheduleAD(state0, model, schedule, 'NonLinearSolver', nls);
its = rep.Iterations;
nmini = cellfun(@(x) numel(x.StepReports), rep.ControlstepReports);
p = cell2mat(cellfun(@(st) st.pressure', states, 'UniformOutput', false));
s = cell2mat(cellfun(@(st) st.s(:)', states, 'UniformOutput', false));
bhp = cell2mat(cellfun(@(w) [w.bhp], ws, 'UniformOutput', false));
qWs = cell2mat(cellfun(@(w) [w.qWs], ws, 'UniformOutput', false));
qOs = cell2mat(cellfun(@(w) [w.qOs], ws, 'UniformOutput', false));
p0 = state0.pressure; s0 = state0.s;
% AMGCL_CPRSolverAD's preparation of the first Newton system
[~, forces] = model.getDrivingForces(schedule.control(1));
m1 = model.validateModel(forces);
st0 = m1.validateState(state0);
[m1, st0] = m1.updateForChangedControls(st0, forces);
[m1, st] = m1.prepareReportstep(st0, st0, dt(1), forces);
[m1, st] = m1.prepareTimestep(st, st0, dt(1), forces);
problem = m1.getEquations(st0, st, dt(1), forces, 'iteration', 1);
cap = CaptureAMGCL_CPRSolverAD('tolerance', 1e-3, 'block_size', ncomp, ...
                               'coarsening', 'aggregation', 'relaxation', 'ilu0');
[dx_cpr, result_cpr] = cap.solveLinearProblem(problem, m1);
A_cpr = cap.A_captured; b_cpr = cap.b_captured; w_cpr = cap.weights_captured;
pressureScaling = cap.pressureScaling;
setup = cap.amgcl_setup;
dx_cpr = cell2mat(cellfun(@(x) x(:), dx_cpr(:), 'UniformOutput', false));
save(fullfile(here, 'ref_blackoil_tutorial_mex_acceleration.mat'), '-v7', 'dims', 'its', 'nmini', ...
     'p', 's', 'bhp', 'qWs', 'qOs', 'p0', 's0', 'A_cpr', 'b_cpr', 'w_cpr', 'dx_cpr', ...
     'result_cpr', 'pressureScaling', 'setup');
fprintf('run: %d iterations; CPR system %d x %d\n', sum(its), size(A_cpr, 1), size(A_cpr, 2));
