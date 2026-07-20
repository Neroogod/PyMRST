%% Dump Phase 41 reference: boundary conditions and sources for AD models.
%
% Run in MATLAB after starting MRST. Writes ref_p41_bc.mat.
%
% Three cases through MRST's GenericBlackOilModel, all with **no wells**:
%
%   1. pressure BCs: water in at 250 bar on one side, out at 150 bar on the
%      other - a model driven entirely by its boundary;
%   2. a flux BC: a surface rate imposed on the inflow face, pressure held on
%      the outflow face;
%   3. a source and a sink, no boundary at all.
%
% PyMRST could not run any of these: its AD models took wells and nothing else.

mrstModule add ad-core ad-props ad-blackoil

gravity off;
thisDir = fileparts(mfilename('fullpath'));

nx = 8;
G  = computeGeometry(cartGrid([nx, 1, 1], [80, 10, 10]));
nc = G.cells.num;
rock = makeRock(G, 100 * milli * darcy, 0.25);

pRef = 200 * barsa;
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',   [1, 5] * centi * poise, ...
                           'rho',  [1000, 800], ...
                           'n',    [2, 2], ...
                           'c',    [1e-6, 1e-4] / barsa, ...
                           'pRef', pRef);

model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                             'oil', true, 'gas', false);
model.toleranceCNV       = 1e-11;
model.toleranceMB        = 1e-14;
model.nonlinearTolerance = 1e-11;

flux_rate = 1e-4;      % surface m3/s in
src_rate  = 1e-4;

for icase = 1 : 3
    bc  = [];
    src = [];
    switch icase
      case 1     % pressure in / pressure out
        bc = pside([], G, 'xmin', 250 * barsa, 'sat', [1, 0]);
        bc = pside(bc, G, 'xmax', 150 * barsa, 'sat', [0, 1]);
        dt = 5 * day * ones(10, 1);
      case 2     % flux in / pressure out
        bc = fluxside([], G, 'xmin',  flux_rate, 'sat', [1, 0]);
        bc = pside(   bc, G, 'xmax', 200 * barsa, 'sat', [0, 1]);
        dt = 5 * day * ones(10, 1);
      case 3     % source and sink, no boundary
        src = addSource([], 1,  src_rate, 'sat', [1, 0]);
        src = addSource(src, nc, -src_rate, 'sat', [0, 1]);
        dt = 5 * day * ones(4, 1);
    end

    schedule.step.val     = dt;
    schedule.step.control = ones(numel(dt), 1);
    schedule.control      = struct('W', [], 'bc', bc, 'src', src);

    state0 = initResSol(G, pRef, [0.2, 0.8]);
    [~, states] = simulateScheduleAD(state0, model, schedule);

    ns = numel(states);
    p_hist  = zeros(ns, nc);
    sw_hist = zeros(ns, nc);
    for i = 1 : ns
        p_hist(i, :)  = states{i}.pressure(:)';
        sw_hist(i, :) = states{i}.s(:, 1)';
    end

    switch icase
      case 1, p1 = p_hist; s1 = sw_hist;   %#ok<NASGU>
      case 2, p2 = p_hist; s2 = sw_hist;   %#ok<NASGU>
      case 3, p3 = p_hist; s3 = sw_hist;   %#ok<NASGU>
    end
    clear schedule;
end

save(fullfile(thisDir, 'ref_p41_bc.mat'), '-v7', ...
     'p1', 's1', 'p2', 's2', 'p3', 's3', ...
     'pRef', 'nx', 'flux_rate', 'src_rate');

fprintf('Wrote ref_p41_bc.mat to %s\n', thisDir);
fprintf('  pressure BC : %.1f -> %.1f bar, sW %.3f -> %.3f\n', ...
        p1(end,1)/barsa, p1(end,end)/barsa, s1(end,1), s1(end,end));
fprintf('  flux BC     : %.1f -> %.1f bar, sW %.3f -> %.3f\n', ...
        p2(end,1)/barsa, p2(end,end)/barsa, s2(end,1), s2(end,end));
fprintf('  source/sink : %.1f -> %.1f bar, sW %.3f -> %.3f\n', ...
        p3(end,1)/barsa, p3(end,end)/barsa, s3(end,1), s3(end,end));
