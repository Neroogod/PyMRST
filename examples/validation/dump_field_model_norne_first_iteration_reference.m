% fieldModelNorneExample in MRST 2024a: the model check for the port. The
% model and settings of dump_field_model_norne_example_reference.m (useMex
% false); saves the equilibrated initial state (with the SWATINIT pc scale),
% the residuals of the first linearised system, and the state after the first
% Newton iteration with a direct linear solver.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat mrst-gui linearsolvers
gravity reset on
useMex = false;
[deck, output] = getNorneFieldDeckOPMData();
egrid = readEclipseOutputFileUnFmt([output.opm.location, '.EGRID']);
init = readEclipseOutputFileUnFmt([output.opm.location, '.INIT']);
[Ge, rock_ecl, Ne, Te] = initGridFromEclipseOutput(init, egrid, 'outputSimGrid', true);
G_sim = Ge{2};
rock = initEclipseRock(deck);
rock = compressRock(rock, G_sim.cells.indexMap);
fluid = initDeckADIFluid(deck, 'useMex', useMex);
model = GenericBlackOilModel(G_sim, [], fluid, 'disgas', true, 'vapoil', true, 'inputdata', deck);
model.rock = rock;
model.operators = setupOperatorsTPFA(G_sim, rock, 'neighbors', Ne, 'trans', Te);
[state0, model, schedule, nls] = initEclipseProblemAD(deck, 'model', model, ...
    'TimestepStrategy', 'ds', 'useCPR', true, 'useMex', useMex);
nls.LinearSolver.tolerance = 1e-3;
nls.maxIterations = 12;
nls.timeStepSelector = FactorTimeStepSelector();
model.toleranceCNV = 1e-2;
model.toleranceMB = 1e-7;
model.FacilityModel = GenericFacilityModel(model);
model.FacilityModel.toleranceWellBHP = 1e-3;
model.FacilityModel.toleranceWellRate = 5e-3;
model.FlowDiscretization = [];
model.FlowPropertyFunctions = [];
model.PVTPropertyFunctions = [];
model.AutoDiffBackend.useMex = useMex;
model.AutoDiffBackend.rowMajor = true;
model = model.validateModel();
xflow = WellComponentTotalVolumeBalanceCrossflow(model);
xflow.onlyLocalDerivatives = true;
model.FacilityModel.FacilityFlowDiscretization.PhaseFlux.allowCrossFlow = false;
model.FacilityModel.FacilityFlowDiscretization.ComponentTotalFlux = xflow;
model.PVTPropertyFunctions.Viscosity.useSaturatedFlag = false;
model.PVTPropertyFunctions.ShrinkageFactors.useSaturatedFlag = false;
RvMax = model.PVTPropertyFunctions.getStateFunction('RvMax');
RvMax.rvReduction = 0.5;
model.PVTPropertyFunctions = model.PVTPropertyFunctions.setStateFunction('RvMax', RvMax);
model.dsMaxAbs = 0.2;
model.dpMaxRel = 0.1;
nls.useRelaxation = true;
nls.minRelaxation = 0.5;
nls.maxIterations = 18;
nls.maxTimestepCuts = 10;
nls.oscillationThreshold = 0.5;
nls.acceptanceFactor = 1;
nls.verbose = false;
nls.LinearSolver.verbose = false;
lsolver = class(nls.LinearSolver);
p0 = state0.pressure; s0 = state0.s; rs0 = state0.rs; rv0 = state0.rv;
pcowScale = state0.pcowScale;
[~, forces] = model.getDrivingForces(schedule.control(1));
dt = schedule.step.val(1);
model = model.validateModel(forces);
state = model.validateState(state0);
s0v = state;
[model, state] = model.prepareReportstep(state, s0v, dt, forces);
[model, state] = model.prepareTimestep(state, s0v, dt, forces);
[problem, ~] = model.getEquations(s0v, state, dt, forces, 'iteration', 1);
res = cellfun(@(x) full(value(x)), problem.equations, 'UniformOutput', false);
eqnames = problem.equationNames;
[state1, ~] = model.stepFunction(state, s0v, dt, forces, BackslashSolverAD(), nls, 1);
p1 = state1.pressure; s1 = state1.s; rs1 = state1.rs; rv1 = state1.rv; bhp1 = [state1.wellSol.bhp];
save(fullfile(here, 'ref_field_model_norne_first_iteration.mat'), '-v7', 'p0', 's0', 'rs0', 'rv0', ...
     'pcowScale', 'res', 'eqnames', 'p1', 's1', 'rs1', 'rv1', 'bhp1');
fprintf('Wrote ref_field_model_norne_first_iteration.mat\n');
