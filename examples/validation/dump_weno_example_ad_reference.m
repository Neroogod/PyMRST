% Reference for wenoExampleAD (validate_ad_core_weno_example_ad.py): the
% script's computations without its figures - the four runs (FIM and AIM,
% single-point upwind and WENO) and the saturation CFL of the first state.
% Per run: iterations and ministeps per report step, the producer's and
% injector's well curves, and the water saturation every tenth step.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
mrstSettings('set', 'useMEX', false);
gravity reset off
nx = 50; ny = 50;
G = computeGeometry(cartGrid([nx, ny], [100, 100]));
rock = makeRock(G, 0.1*darcy, 0.3);
fluid = initSimpleADIFluid('n', [1, 1], 'mu', [1, 1]*centi*poise, 'rho', [100, 100], 'phases', 'wo');
model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, 'gas', false);
time = 1*year;
irate = 2*sum(model.operators.pv)/time;
W = [];
W = addWell(W, G, rock, 1, 'comp_i', [1, 0], 'val', irate, 'type', 'rate');
W = addWell(W, G, rock, G.cells.num, 'comp_i', [1, 0], 'val', 50*barsa, 'type', 'bhp');
n = 300;
dt = repmat(time/n, n, 1);
schedule = simpleSchedule(dt, 'W', W);
state0 = initResSol(G, 100*barsa, [0, 1]);
[ws, states, report] = simulateScheduleAD(state0, model, schedule);
cfl = estimateSaturationCFL(model, states{1}, max(schedule.step.val), 'forces', schedule.control(1));
model_weno = setWENODiscretization(model);
[ws_weno, states_weno, report_weno] = simulateScheduleAD(state0, model_weno, schedule);
model_e = setTimeDiscretization(model, 'AIM', 'initialstep', 0.02*day, 'verbose', true);
[ws_e, states_e, report_e] = simulateScheduleAD(state0, model_e, schedule);
model_weno_expl = setTimeDiscretization(model_weno, 'AIM', 'initialstep', 0.02*day, 'verbose', true);
[ws_weno_ex, states_weno_ex, report_weno_ex] = simulateScheduleAD(state0, model_weno_expl, schedule);
out = struct('cfl', cfl);
names = {'fim', 'weno', 'aim', 'aim_weno'};
runs = {{ws, states, report}, {ws_weno, states_weno, report_weno}, ...
        {ws_e, states_e, report_e}, {ws_weno_ex, states_weno_ex, report_weno_ex}};
keep = [1:10:n, n];
for r = 1:4
    [w, st, rep] = deal(runs{r}{:});
    o = struct();
    o.its = cellfun(@(x) x.Iterations, rep.ControlstepReports);
    o.nmini = cellfun(@(x) numel(x.StepReports), rep.ControlstepReports);
    o.dt = cell2mat(cellfun(@(x) cellfun(@(y) y.Timestep, x.StepReports), ...
                    rep.ControlstepReports, 'UniformOutput', false));
    o.qWs = cell2mat(cellfun(@(x) [x.qWs], w, 'UniformOutput', false));
    o.qOs = cell2mat(cellfun(@(x) [x.qOs], w, 'UniformOutput', false));
    o.bhp = cell2mat(cellfun(@(x) [x.bhp], w, 'UniformOutput', false));
    o.sw = cell2mat(cellfun(@(x) x.s(:, 1)', st(keep), 'UniformOutput', false));
    o.p = cell2mat(cellfun(@(x) x.pressure', st(keep), 'UniformOutput', false));
    out.(names{r}) = o;
end
out.keep = keep;
save(fullfile(here, 'ref_weno_example_ad.mat'), '-struct', 'out', '-v7');
