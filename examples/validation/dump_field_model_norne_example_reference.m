% fieldModelNorneExample in MRST 2024a without its figures, on its first
% report steps: the model the script builds (ECLIPSE's grid and
% transmissibilities, hysteresis, threshold pressures, the volume-balanced
% crossflow), with useMex false (MRST's MEX does not build here). Two runs:
% the first two report steps with a direct linear solver (BackslashSolverAD),
% which checks the model, and the first five with the script's CPR/GMRES-ILU.
% Saves the initial state and, per report step, the well solutions, the
% Newton iterations and the states.
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

out = struct();
cpr = nls.LinearSolver;   % NonLinearSolver is a handle: restore it explicitly
for run_ = {'direct', 'iterative'}
    r = run_{1};
    solver = nls;
    if strcmp(r, 'direct')
        n = 2; solver.LinearSolver = BackslashSolverAD();
    else
        n = 5; solver.LinearSolver = cpr;
    end
    sch = schedule;
    sch.step.val = sch.step.val(1:n);
    sch.step.control = sch.step.control(1:n);
    solver.timeStepSelector = FactorTimeStepSelector();
    t = tic;
    [ws, states, rep] = simulateScheduleAD(state0, model, sch, 'NonLinearSolver', solver);
    res = struct('walltime', toc(t), 'its', rep.Iterations, 'lsolver', class(solver.LinearSolver), ...
                 'nmini', cellfun(@(x) numel(x.StepReports), rep.ControlstepReports));
    nw = numel(ws{1});
    [res.bhp, res.qWs, res.qOs, res.qGs] = deal(zeros(n, nw));
    for k = 1:n
        res.bhp(k, :) = [ws{k}.bhp]; res.qWs(k, :) = [ws{k}.qWs];
        res.qOs(k, :) = [ws{k}.qOs]; res.qGs(k, :) = [ws{k}.qGs];
    end
    res.status = cell2mat(cellfun(@(w) [w.status], ws, 'UniformOutput', false));
    res.p = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
    res.sw = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
    res.sg = cell2mat(cellfun(@(s) s.s(:, 3)', states, 'UniformOutput', false));
    res.rs = cell2mat(cellfun(@(s) s.rs', states, 'UniformOutput', false));
    res.rv = cell2mat(cellfun(@(s) s.rv', states, 'UniformOutput', false));
    out.(r) = res;
    fprintf('%s: %d steps, %d iterations, %.0f s\n', r, n, sum(res.its), res.walltime);
end
names = {ws{1}.name};
direct = out.direct; iterative = out.iterative;
save(fullfile(here, 'ref_field_model_norne_example.mat'), '-v7', 'direct', 'iterative', ...
     'p0', 's0', 'rs0', 'rv0', 'lsolver', 'names');
fprintf('Wrote ref_field_model_norne_example.mat\n');
