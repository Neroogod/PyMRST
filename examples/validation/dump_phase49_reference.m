%% Dump Phase 49 reference: a MULTI-PERFORATION well, from MRST ad-blackoil.
%
% Run in MATLAB after starting MRST. Writes ref_p49_multiperf.mat.
%
% Every well in every MATLAB comparison this port has ever run has had **one
% perforation**. SPE 1: one. Norne: one. The oil-water quarter spot, the
% gas-condensate column, the aquifer case: one each. So the part of the well
% model that only exists when a well has *more* than one perforation - the
% wellbore column, accumulated segment by segment down the hole, each segment
% weighing the cumulative stream that flows past it - has never once been
% checked against MRST.
%
% It is not a corner case. Every real well has dozens of perforations, and the
% column is worth bar, not millibar.
%
% So: two genuinely vertical wells, five perforations each, gravity on, and a
% deliberately heavy contrast between the phases (water 1050, oil 700) so that
% *which* fluid is in *which* segment actually matters. If the column were
% computed with a single density over the whole depth - which is what PyMRST
% used to do, and is exactly right for a one-cell well - it would disagree.
%
% The schedule is chosen so MRST takes exactly one ministep per report step
% (asserted below), so the comparison is against a single-step solve rather
% than against MRST's timestep controller.

mrstModule add ad-core ad-props ad-blackoil

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

% -- a column of cells, so a vertical well can perforate several of them ------
pRef = 250*barsa;
nx = 8; ny = 1; nz = 5;
G = cartGrid([nx, ny, nz], [400, 100, 100]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + 2000;   % 2000 - 2100 m
G = computeGeometry(G);
nc = G.cells.num;
rock = makeRock(G, 200*milli*darcy, 0.25);

% Heavy density contrast: which phase is in which wellbore segment must matter.
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',  [0.4, 3.0]*centi*poise, ...
                           'rho', [1050, 700], ...
                           'n',   [2, 2], ...
                           'c',   [4.0e-5, 1.2e-4]/barsa, ...
                           'pRef', pRef);

model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                             'gas', false);
model.toleranceCNV = 1e-10;      % converge hard: MRST's 1e-3 default leaves a
model.toleranceMB  = 1e-13;      % residual of its own and muddies the compare
model = model.validateModel();

% -- two vertical wells, FIVE perforations each -------------------------------
% verticalWell picks the whole column, which is the point.
W = verticalWell([], G, rock, 1, 1, (1:nz), 'Type', 'rate', ...
                 'Val', 4e-3, 'Comp_i', [1, 0], 'sign', 1, ...
                 'Radius', 0.1, 'Name', 'INJ');
W = verticalWell(W, G, rock, nx, 1, (1:nz), 'Type', 'bhp', ...
                 'Val', 200*barsa, 'Comp_i', [0, 1], 'sign', -1, ...
                 'Radius', 0.1, 'Name', 'PROD');

assert(numel(W(1).cells) == nz && numel(W(2).cells) == nz, ...
       'both wells must be multi-perforation, or this tests nothing');

nstep = 15;
dt = repmat(20*day, nstep, 1);
schedule = simpleSchedule(dt, 'W', W);

state0 = initResSol(G, pRef, [0.15, 0.85]);

[wellSols, states, report] = simulateScheduleAD(state0, model, schedule);

nmini = zeros(1, numel(report.ControlstepReports));
for k = 1:numel(report.ControlstepReports)
    nmini(k) = numel(report.ControlstepReports{k}.StepReports);
end
assert(all(nmini == 1), ...
    'MRST sub-divided a report step; the comparison would be meaningless.');

p_hist  = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sw_hist = cell2mat(cellfun(@(s) s.s(:,1)',   states, 'UniformOutput', false));

% The wellbore column itself - the thing being tested.
cdp_inj  = cell2mat(cellfun(@(w) w(1).cdp(:)', wellSols, 'UniformOutput', false));
cdp_prod = cell2mat(cellfun(@(w) w(2).cdp(:)', wellSols, 'UniformOutput', false));

qWs = cell2mat(cellfun(@(w) [w.qWs], wellSols, 'UniformOutput', false));
qOs = cell2mat(cellfun(@(w) [w.qOs], wellSols, 'UniformOutput', false));
bhp = cell2mat(cellfun(@(w) [w.bhp], wellSols, 'UniformOutput', false));

inj_cells  = W(1).cells;   inj_WI  = W(1).WI;   inj_dZ  = W(1).dZ;
prod_cells = W(2).cells;   prod_WI = W(2).WI;   prod_dZ = W(2).dZ;
inj_ref  = W(1).refDepth;  prod_ref = W(2).refDepth;

save(fullfile(thisDir, 'ref_p49_multiperf.mat'), '-v7', ...
     'p_hist', 'sw_hist', 'cdp_inj', 'cdp_prod', 'qWs', 'qOs', 'bhp', ...
     'inj_cells', 'inj_WI', 'inj_dZ', 'inj_ref', ...
     'prod_cells', 'prod_WI', 'prod_dZ', 'prod_ref', ...
     'nx', 'ny', 'nz', 'nstep', 'pRef');

fprintf('Wrote ref_p49_multiperf.mat to %s\n', thisDir);
fprintf('  perforations: INJ %d, PROD %d\n', numel(inj_cells), numel(prod_cells));
fprintf('  ministeps per report step: all %d\n', max(nmini));
fprintf('  INJ  cdp (final, bar): %s\n', mat2str(round(cdp_inj(end,:)/barsa, 3)));
fprintf('  PROD cdp (final, bar): %s\n', mat2str(round(cdp_prod(end,:)/barsa, 3)));
fprintf('  a single-density column would give equal increments; these do not\n');
