function dump_ad_blackoil_egg_reference()
% MRST fixture for eggExample setup and first production report.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrst = fullfile(fileparts(root), 'mrst-2024a-Geomech');
cd(mrst); startup; mrstModule add ad-core ad-blackoil deckformat test-suite;
[G, rock, fluid, deck, state0] = setupEGG('realization', 0);
[state, model, schedule, nonlinear] = initEclipseProblemAD(deck, 'G', G, 'TimestepStrategy', 'none');
one = schedule; one.step.val = schedule.step.val(1); one.step.control = schedule.step.control(1);
[wellSols, states] = simulateScheduleAD(state, model, one, 'NonLinearSolver', nonlinear);
nc = G.cells.num; cartDims = G.cartDims; indexMap = G.cells.indexMap;
centroids = G.cells.centroids; poro = rock.poro; perm = rock.perm;
pressure0 = state.pressure; sw0 = state.s(:, 1);
step_val = schedule.step.val; step_control = schedule.step.control;
well_names = {schedule.control(1).W.name};
well_cells = arrayfun(@(w) numel(w.cells), schedule.control(1).W);
well_sign = [schedule.control(1).W.sign]; well_val = [schedule.control(1).W.val];
pressure1 = states{1}.pressure; sw1 = states{1}.s(:, 1);
bhp1 = [wellSols{1}.bhp]'; qWs1 = [wellSols{1}.qWs]'; qOs1 = [wellSols{1}.qOs]';
out = fullfile(root, 'examples', 'validation', 'ref_ad_blackoil_egg.mat');
save(out, 'nc', 'cartDims', 'indexMap', 'centroids', 'poro', 'perm', ...
    'pressure0', 'sw0', 'step_val', 'step_control', 'well_names', ...
    'well_cells', 'well_sign', 'well_val', 'pressure1', 'sw1', 'bhp1', ...
    'qWs1', 'qOs1', '-v7');
fprintf('Saved %s\n', out);
end
