% Reference for setupSPE10_AD (validate_spe10_setup_spe10_ad.py): layers 1 and
% 85 with rows 1:110 as 2D models (as compositeModelTutorial's TestCase
% 'spe10_wo' builds them, gravity reset on). Wells (cells, indices, controls),
% pore volumes, transmissibilities, time steps, and the first five steps
% simulated: iterations and well curves.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props spe10
mrstSettings('set', 'useMEX', false);
out = struct();
layers = [1, 85];
for k = 1:2
    gravity reset on
    [state0, model, schedule] = setupSPE10_AD('layers', layers(k), 'J', 1:110, 'make2D', true);
    W = schedule.control(1).W;
    o = struct('pv', model.operators.pv, 'T', model.operators.T, 'dt', schedule.step.val, ...
               'cells', vertcat(W.cells), 'WI', vertcat(W.WI), 'val', [W.val], ...
               'p0', state0.pressure, 's0', state0.s);
    sub = schedule; sub.step.val = sub.step.val(1:5); sub.step.control = sub.step.control(1:5);
    [ws, states, rep] = simulateScheduleAD(state0, model, sub);
    o.its = cellfun(@(x) x.Iterations, rep.ControlstepReports);
    o.bhp = cell2mat(cellfun(@(x) [x.bhp], ws, 'UniformOutput', false));
    o.qWs = cell2mat(cellfun(@(x) [x.qWs], ws, 'UniformOutput', false));
    o.qOs = cell2mat(cellfun(@(x) [x.qOs], ws, 'UniformOutput', false));
    o.p = states{end}.pressure; o.sw = states{end}.s(:, 1);
    out.(sprintf('layer%d', layers(k))) = o;
end
save(fullfile(here, 'ref_setup_spe10_ad.mat'), '-struct', 'out', '-v7');
