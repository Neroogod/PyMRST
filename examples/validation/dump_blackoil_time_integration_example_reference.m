% blackoilTimeIntegrationExample in MRST 2024a without its figures, for
% name = 'spe1' (the script's default): the three packed problems simulated
% (validate_ad_oo_blackoil_time_integration_example.py). Their old outputs are
% cleared first so the run is the script's, not a cache.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat ad-props linearsolvers test-suite
mrstVerbose on
name = 'spe1';
[G, rock, fluid, deck, state] = setupSPE1();
[state0, model, schedule, nls] = initEclipseProblemAD(deck, 'TimestepStrategy', 'none');
implicit = packSimulationProblem(state0, model, schedule, name, ...
    'NonLinearSolver', nls, 'Name', 'Fully-Implicit');
model_explicit = setTimeDiscretization(model, 'explicit');
explicit = packSimulationProblem(state0, model_explicit, schedule, name, ...
    'NonLinearSolver', nls, 'Name', 'Explicit');
model_aim = setTimeDiscretization(model, 'adaptive-implicit');
aim = packSimulationProblem(state0, model_aim, schedule, name,...
    'NonLinearSolver', nls, 'Name', 'Adaptive-Implicit (AIM)');
problems = {implicit, explicit, aim};
for i = 1:numel(problems)
    clearPackedSimulatorOutput(problems{i}, 'prompt', false);
end
t = tic;
simulatePackedProblem(problems);
walltime = toc(t);
[ws, states, reports, names, T] = getMultiplePackedSimulatorOutputs(problems);
out = struct();
keys = {'fim', 'explicit', 'aim'};
for i = 1:3
    r = struct();
    r.name = names{i};
    r.its = cellfun(@(x) x.Iterations, reports{i});
    r.nmini = cellfun(@(x) numel(x.StepReports), reports{i});
    r.dt = getReportMinisteps(reports{i});
    r.p = cell2mat(cellfun(@(st) st.pressure', states{i}, 'UniformOutput', false));
    r.sw = cell2mat(cellfun(@(st) st.s(:, 1)', states{i}, 'UniformOutput', false));
    r.sg = cell2mat(cellfun(@(st) st.s(:, 3)', states{i}, 'UniformOutput', false));
    r.rs = cell2mat(cellfun(@(st) st.rs', states{i}, 'UniformOutput', false));
    r.bhp = cell2mat(cellfun(@(w) [w.bhp], ws{i}, 'UniformOutput', false));
    r.qOs = cell2mat(cellfun(@(w) [w.qOs], ws{i}, 'UniformOutput', false));
    r.qGs = cell2mat(cellfun(@(w) [w.qGs], ws{i}, 'UniformOutput', false));
    r.qWs = cell2mat(cellfun(@(w) [w.qWs], ws{i}, 'UniformOutput', false));
    r.T = T{i};
    if i == 3
        r.imp_frac = cellfun(@(x) sum(x.implicit)/numel(x.implicit), states{3});
    end
    out.(keys{i}) = r;
    fprintf('%s: %d steps, %d iterations, %d ministeps\n', names{i}, numel(r.its), sum(r.its), sum(r.nmini));
end
fim = out.fim; explicit = out.explicit; aim = out.aim;
save(fullfile(here, 'ref_blackoil_time_integration_example.mat'), '-v7', 'fim', 'explicit', 'aim', 'walltime');
fprintf('walltime %.0f s\n', walltime);
