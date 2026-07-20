%% Dump Phase 38 reference: capillary pressure, from MRST's ad-core graph.
%
% Run in MATLAB after starting MRST. Writes ref_p38_pc.mat.
%
% The same oil-water case is run twice - once with a capillary pressure and
% once without - through MRST's *state-function* model (GenericBlackOilModel),
% which is the one whose BlackOilCapillaryPressure / PhasePressures /
% PressureGradient chain PyMRST now mirrors.
%
% Running both matters: the no-pc case must reproduce what PyMRST already
% matched before capillarity existed (so nothing regressed), and the pc case
% must exercise the new chain.

mrstModule add ad-core ad-props ad-blackoil

gravity off;
thisDir = fileparts(mfilename('fullpath'));

nx = 15;
G = computeGeometry(cartGrid([nx, nx, 1], [300, 300, 10]));
nc = G.cells.num;

kvals = (50 + mod(11 * (0 : nc - 1)', 200)) * milli * darcy;
rock  = makeRock(G, kvals, 0.25);

pRef = 200 * barsa;
pc0  = 5 * barsa;            % strong enough that it changes the answer

for usePc = [false, true]
    fluid = initSimpleADIFluid('phases', 'WO', ...
                               'mu',   [1, 5] * centi * poise, ...
                               'rho',  [1000, 800], ...
                               'n',    [2, 2], ...
                               'c',    [1e-6, 1e-4] / barsa, ...
                               'pRef', pRef);
    if usePc
        % pcOW = p_oil - p_water, decreasing in water saturation.
        fluid.pcOW = @(sw) pc0 * (1 - sw);
    end

    model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                                 'oil', true, 'gas', false);
    model.toleranceCNV       = 1e-11;
    model.toleranceMB        = 1e-13;
    model.nonlinearTolerance = 1e-11;

    W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', 2e-3, ...
                'Comp_i', [1, 0], 'Sign', 1, 'Radius', 0.1);
    W = addWell(W, G, rock, nc, 'Type', 'bhp', 'Val', 150 * barsa, ...
                'Comp_i', [0, 1], 'Sign', -1, 'Radius', 0.1);

    dt = 30 * day * ones(8, 1);
    schedule.step.val     = dt;
    schedule.step.control = ones(numel(dt), 1);
    schedule.control      = struct('W', W);

    state0 = initResSol(G, pRef, [0.2, 0.8]);

    [~, states] = simulateScheduleAD(state0, model, schedule);

    ns = numel(states);
    p_hist  = zeros(ns, nc);
    sw_hist = zeros(ns, nc);
    bhp     = zeros(ns, numel(W));
    for i = 1 : ns
        p_hist(i, :)  = states{i}.pressure(:)';
        sw_hist(i, :) = states{i}.s(:, 1)';
        bhp(i, :)     = [states{i}.wellSol.bhp];
    end

    if usePc
        pc_p = p_hist;  pc_sw = sw_hist;  pc_bhp = bhp;   %#ok<NASGU>
    else
        no_p = p_hist;  no_sw = sw_hist;  no_bhp = bhp;   %#ok<NASGU>
    end
    clear schedule W;
end

save(fullfile(thisDir, 'ref_p38_pc.mat'), '-v7', ...
     'pc_p', 'pc_sw', 'pc_bhp', 'no_p', 'no_sw', 'no_bhp', ...
     'pRef', 'pc0', 'nx', 'kvals');

fprintf('Wrote ref_p38_pc.mat to %s\n', thisDir);
fprintf('  capillary pressure changes sW by up to %g saturation units\n', ...
        max(abs(pc_sw(:) - no_sw(:))));
