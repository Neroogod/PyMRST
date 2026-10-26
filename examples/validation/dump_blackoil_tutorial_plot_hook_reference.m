% blackoilTutorialPlotHook in MRST 2024a without its after-step function
% (getPlotAfterStep only draws the well curves and the runtime panel; it
% does not change the run): SPE1 with selectLinearSolverAD's solver.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat test-suite
[G, rock, fluid, deck, state0] = setupSPE1();
model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);
linsolve = selectLinearSolverAD(model);
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule, ...
    'Verbose', true, 'linearsolver', linsolve);
lsolver = class(linsolve);
p = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sat = cell2mat(cellfun(@(s) reshape(s.s, 1, []), states, 'UniformOutput', false));
rs = cell2mat(cellfun(@(s) s.rs', states, 'UniformOutput', false));
[qWs, qOs, qGs, bhp] = wellSolToVector(wellSols);
its = report.Iterations;
save(fullfile(here, 'ref_blackoil_tutorial_plot_hook.mat'), 'lsolver', 'p', 'sat', 'rs', ...
     'qWs', 'qOs', 'qGs', 'bhp', 'its', '-v7');
