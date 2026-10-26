% simulateSPE1 and simulateSPE3, as published, without the plots; saves the
% packed problems' output (states, well solutions, reports).
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-blackoil ad-core ad-props deckformat
for c = {'spe1', 'spe3'}
    name = c{1};
    pth = getDatasetPath(name);
    fn  = fullfile(pth, ['BENCH_', upper(name), '.DATA']);
    problem = initEclipsePackedProblemAD(fn);
    simulatePackedProblem(problem, 'restartStep', 1);
    [ws, states, reports] = getPackedSimulatorOutput(problem);
    grab = @(f) cell2mat(cellfun(@(s) s.(f)', states, 'UniformOutput', false));
    p = grab('pressure');
    s = cell2mat(cellfun(@(s) reshape(s.s, 1, []), states, 'UniformOutput', false));
    [qWs, qOs, qGs, bhp] = wellSolToVector(ws);
    its = cellfun(@(r) r.Iterations, reports);
    nmini = cellfun(@(r) numel(r.StepReports), reports);
    names = {ws{1}.name};
    basename = problem.BaseName;
    lsolver = class(problem.SimulatorSetup.NonLinearSolver.LinearSolver);
    backend = class(problem.SimulatorSetup.model.AutoDiffBackend);
    if isfield(states{1}, 'rs'), rs = grab('rs'); else, rs = []; end
    if isfield(states{1}, 'rv'), rv = grab('rv'); else, rv = []; end
    save(fullfile(here, ['ref_simulate_', name, '.mat']), 'p', 's', 'rs', 'rv', ...
         'qWs', 'qOs', 'qGs', 'bhp', 'its', 'nmini', 'names', 'basename', 'lsolver', ...
         'backend', '-v7');
end
