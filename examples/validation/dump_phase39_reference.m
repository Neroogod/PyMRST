%% Dump Phase 39 reference: well control switching, from MRST's ad-core.
%
% Run in MATLAB after starting MRST. Writes ref_p39_limits.mat.
%
% A rate-controlled injector with a bottom-hole pressure limit it cannot meet,
% and a BHP-controlled producer with a liquid-rate limit it exceeds. Both must
% switch, and *when* they switch (and to what) is what this pins - the
% mechanism is easy, the switching point is not.
%
% The same case is also run with the limits set unreachably far away, so the
% no-switch path is checked too: a limit that is never hit must leave the well
% exactly as it was.

mrstModule add ad-core ad-props ad-blackoil

gravity off;
thisDir = fileparts(mfilename('fullpath'));

nx = 10;
G  = computeGeometry(cartGrid([nx, nx, 1], [200, 200, 10]));
nc = G.cells.num;
rock = makeRock(G, 100 * milli * darcy, 0.25);

pRef = 200 * barsa;
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',   [1, 5] * centi * poise, ...
                           'rho',  [1000, 800], ...
                           'n',    [2, 2], ...
                           'c',    [1e-6, 1e-4] / barsa, ...
                           'pRef', pRef);
fluid.rhoSinj = [1000, 800];    % this tree's local SimpleWell modification

% At 1e-3 m3/s this injector needs ~340 bar, so a 300 bar limit lets it run on
% rate for the first steps and switch part-way through - which tests *when* it
% switches, not merely that it can. The producer would pull ~1e-3 m3/s of
% liquid at 150 bar, so a -6e-4 limit bites too.
q_inj    = 1e-3;
bhp_lim  = 300 * barsa;
bhp_prod = 150 * barsa;
lrat_lim = -6e-4;               % negative: production

for tight = [true, false]
    if tight
        injLims  = struct('bhp',  bhp_lim);
        prodLims = struct('lrat', lrat_lim);
    else                        % limits placed far out of reach
        injLims  = struct('bhp',  1e5 * barsa);   % truly unreachable
        prodLims = struct('lrat', -1e3);
    end

    W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', q_inj, ...
                'Comp_i', [1, 0], 'Sign', 1, 'Radius', 0.1, ...
                'Name', 'inj',  'lims', injLims);
    W = addWell(W, G, rock, nc, 'Type', 'bhp', 'Val', bhp_prod, ...
                'Comp_i', [0, 1], 'Sign', -1, 'Radius', 0.1, ...
                'Name', 'prod', 'lims', prodLims);

    model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                                 'oil', true, 'gas', false);
    model.toleranceCNV       = 1e-11;
    model.toleranceMB        = 1e-13;
    model.nonlinearTolerance = 1e-11;

    dt = 30 * day * ones(6, 1);
    schedule.step.val     = dt;
    schedule.step.control = ones(numel(dt), 1);
    schedule.control      = struct('W', W);

    state0 = initResSol(G, pRef, [0.2, 0.8]);
    [wellSols, states] = simulateScheduleAD(state0, model, schedule);

    ns = numel(states);
    p_hist  = zeros(ns, nc);
    sw_hist = zeros(ns, nc);
    bhp     = zeros(ns, 2);
    qWs     = zeros(ns, 2);
    qOs     = zeros(ns, 2);
    for i = 1 : ns
        p_hist(i, :)  = states{i}.pressure(:)';
        sw_hist(i, :) = states{i}.s(:, 1)';
        bhp(i, :)     = [wellSols{i}.bhp];
        qWs(i, :)     = [wellSols{i}.qWs];
        qOs(i, :)     = [wellSols{i}.qOs];
    end
    % the control mode each well ended each step on - the switch point
    types = cell(ns, 2);
    for i = 1 : ns
        types(i, :) = {wellSols{i}.type};
    end

    if tight
        t_p = p_hist;  t_sw = sw_hist;  t_bhp = bhp;
        t_qw = qWs;    t_qo = qOs;      t_types = types;   %#ok<NASGU>
    else
        f_p = p_hist;  f_sw = sw_hist;  f_bhp = bhp;
        f_qw = qWs;    f_qo = qOs;      f_types = types;   %#ok<NASGU>
    end
    clear schedule W;
end

save(fullfile(thisDir, 'ref_p39_limits.mat'), '-v7', ...
     't_p', 't_sw', 't_bhp', 't_qw', 't_qo', 't_types', ...
     'f_p', 'f_sw', 'f_bhp', 'f_qw', 'f_qo', 'f_types', ...
     'pRef', 'nx', 'q_inj', 'bhp_lim', 'bhp_prod', 'lrat_lim');

fprintf('Wrote ref_p39_limits.mat to %s\n', thisDir);
fprintf('  with limits : injector ended on "%s", producer on "%s"\n', ...
        t_types{end, 1}, t_types{end, 2});
fprintf('  without     : injector ended on "%s", producer on "%s"\n', ...
        f_types{end, 1}, f_types{end, 2});
