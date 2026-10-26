% simulateSPE9 with MRST's non-MEX kernels ('useMex' false, 'rowMajorAD' true;
% the published 'useMex' true needs MRST's C++ kernels, which fail to compile
% on this machine: the compiler rejects the Apple M4 CPU target), and the same
% problem with a direct linear solver; saves both runs' output.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-blackoil ad-core ad-props deckformat
fn = fullfile(getDatasetPath('spe9'), 'BENCH_SPE9.DATA');
for variant = {'published', 'direct'}
    v = variant{1};
    if strcmp(v, 'direct')
        [state0, model, schedule, nls] = initEclipseProblemAD(fn, 'useMex', false, 'rowMajorAD', true);
        nls.LinearSolver = BackslashSolverAD();
        problem = packSimulationProblem(state0, model, schedule, 'SPE9_direct', 'NonLinearSolver', nls);
    else
        problem = initEclipsePackedProblemAD(fn, 'useMex', false, 'rowMajorAD', true);
    end
    t = tic;
    simulatePackedProblem(problem, 'restartStep', 1);
    walltime = toc(t);
    [ws, states, reports] = getPackedSimulatorOutput(problem);
    grab = @(f) cell2mat(cellfun(@(s) s.(f)', states, 'UniformOutput', false));
    p = grab('pressure'); rs = grab('rs');
    s = cell2mat(cellfun(@(s) reshape(s.s, 1, []), states, 'UniformOutput', false));
    [qWs, qOs, qGs, bhp] = wellSolToVector(ws);
    its = cellfun(@(r) r.Iterations, reports);
    nmini = cellfun(@(r) numel(r.StepReports), reports);
    types = cellfun(@(w) strjoin({w.type}, ','), ws, 'UniformOutput', false);
    names = {ws{1}.name};
    lsolver = class(problem.SimulatorSetup.NonLinearSolver.LinearSolver);
    backend = class(problem.SimulatorSetup.model.AutoDiffBackend);
    basename = problem.BaseName;
    save(fullfile(here, ['ref_simulate_spe9_', v, '.mat']), 'p', 'rs', 's', 'qWs', ...
         'qOs', 'qGs', 'bhp', 'its', 'nmini', 'types', 'names', 'lsolver', 'backend', ...
         'basename', 'walltime', '-v7');
end
