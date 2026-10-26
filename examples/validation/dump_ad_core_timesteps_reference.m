% The time-step selectors and rampupTimesteps in MRST 2024a, driven through a
% scripted sequence of control changes, picks, cuts and step reports, for
% validate_ad_core_timesteps.py.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
G = computeGeometry(cartGrid([3, 1, 1]));
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 1]*centi*poise, 'rho', [1000, 800], 'n', [2, 2]);
model = GenericBlackOilModel(G, makeRock(G, 1e-13, 0.3), fluid, 'gas', false);
solver = struct('maxIterations', 25);
its = [6, 9, 12, 4, 25, 7, 5, 8];
conv = [1, 1, 1, 1, 0, 1, 1, 1];
ctrl = [1, 1, 1, 2, 2, 2, 2, 3];
sw = [0.1, 0.15, 0.3, 0.32, 0.32, 0.5, 0.52, 0.6];
selectors = {SimpleTimeStepSelector('firstRampupStepRelative', 0.1, 'firstRampupStep', 2*day, ...
                                    'maxTimestep', 40*day), ...
             IterationCountTimeStepSelector('targetIterationCount', 8, 'firstRampupStepRelative', 0.1), ...
             StateChangeTimeStepSelector('targetProps', {'s'}, 'targetChangeAbs', 0.2, ...
                                         'targetIterationCount', inf), ...
             FactorTimeStepSelector('maxRelativeAdjustment', 1.5, 'resetOnControlsChanged', true)};
dts = zeros(numel(selectors), numel(its));
for k = 1:numel(selectors)
    sel = selectors{k};
    dt_prev = []; state_prev = []; dt = 30*day;
    for i = 1:numel(its)
        if i == 1 || ctrl(i) ~= ctrl(i-1) || mod(i, 3) == 0
            sel.newControlStep(struct('controlId', ctrl(i)));
        end
        state = struct('s', [sw(i)*ones(3, 1), (1-sw(i))*ones(3, 1)], 'pressure', 1e7*ones(3, 1));
        if i > 1 && ~conv(i-1)
            dt = sel.cutTimestep(dt_prev, dt, model, solver, state_prev, state, []);
        else
            dt = sel.pickTimestep(dt_prev, dt, model, solver, state_prev, state, []);
        end
        dts(k, i) = dt;
        sel.storeTimestep(struct('Converged', logical(conv(i)), 'Iterations', its(i), 'Timestep', dt));
        if conv(i)
            dt_prev = dt; state_prev = state;
        end
    end
end
ramp1 = rampupTimesteps(100*day, 10*day);
ramp2 = rampupTimesteps(3*day, 10*day, 4);
ramp3 = rampupTimesteps(105*day, 10*day, 3);
save(fullfile(here, 'ref_ad_core_timesteps.mat'), '-v7', 'dts', 'its', 'conv', 'ctrl', 'sw', 'ramp1', 'ramp2', 'ramp3');
disp(dts/day);
