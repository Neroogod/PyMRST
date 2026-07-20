%% Dump Phase 48b reference: a *coupled* aquifer simulation, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p48b_aquifer.mat.
%
% Phase 48 validated the aquifer *formulas* in isolation, by fixing the inputs
% MRST evaluated and replaying PyMRST's arithmetic on the same numbers. That is
% the right way to test a formula and it leaves one thing untested: the
% **coupling**. Two things only exist inside a running simulation -
%
%   - `addAquifersContribution`, which subtracts the influx from the water
%     equation, so the aquifer enters the *Jacobian* and the reservoir feels it;
%   - `updateAfterConvergence`, which advances the aquifer's own pressure and
%     volume once a step is accepted, so the tank actually drains.
%
% - and a port could get both formulas exactly right while wiring either of them
% up wrongly: with the sign flipped, the aquifer would drain the reservoir
% instead of supporting it; held explicit at the wrong state, it would drain at
% the wrong rate. Neither shows up in Phase 48.
%
% So: a real oil-water reservoir, a producer at one end pulling it down, and an
% aquifer attached to the far end holding it up. Everything the two codes must
% agree on is dumped - the pressure and saturation fields, the well rates, and
% the aquifer's own pressure and volume trajectory.
%
% The schedule is chosen so MRST takes **exactly one ministep per report step**
% (asserted below). If it had to sub-divide, it would be solving a different
% discrete problem from a single-step solve and matching it would mean
% reproducing its timestep controller rather than its physics.

mrstModule add ad-core ad-props ad-blackoil

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

% -- reservoir ---------------------------------------------------------------
pRef = 200*barsa;
nx = 10; nz = 3;
G = cartGrid([nx, 1, nz], [500, 100, 30]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + 2000;
G = computeGeometry(G);
nc = G.cells.num;
rock = makeRock(G, 100*milli*darcy, 0.25);

fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',  [0.5, 2.0]*centi*poise, ...
                           'rho', [1000, 800], ...
                           'n',   [2, 2], ...
                           'c',   [4.0e-5, 1.0e-4]/barsa, ...
                           'pRef', pRef);

model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                             'gas', false);

% -- the aquifer, attached to the far column (i = nx) -------------------------
aquind = struct('aquid', 1, 'conn', 2, 'pvttbl', 3, 'J', 4, 'C', 5, ...
                'alpha', 6, 'depthconn', 7, 'depthaq', 8);

% every cell of the last column, one connection each
conncells = find(G.cells.centroids(:, 1) > 500 - 500/nx - 1e-9);
nconn     = numel(conncells);

Jval    = repmat(4.0e-11, nconn, 1);
Caq     = 1.0e-5/barsa;
alpha   = repmat(1/nconn, nconn, 1);        % equal shares, summing to one
depthaq = 2015;
depthconn = G.cells.centroids(conncells, 3);

aquifers = nan(nconn, 8);
aquifers(:, aquind.aquid)     = 1;
aquifers(:, aquind.conn)      = conncells;
aquifers(:, aquind.pvttbl)    = 1;
aquifers(:, aquind.J)         = Jval;
aquifers(:, aquind.C)         = Caq;
aquifers(:, aquind.alpha)     = alpha;
aquifers(:, aquind.depthconn) = depthconn;
aquifers(:, aquind.depthaq)   = depthaq;

aquiferprops = struct('C', Caq, 'J', sum(Jval), 'depthaq', depthaq, ...
                      'pvttbl', 1);
initvals = struct('pressures', 250*barsa, 'volumes', 5.0e7);

model.AquiferModel = AquiferModel(model, aquifers, aquind, aquiferprops, ...
                                  initvals);

% Converge *hard*. MRST's defaults (toleranceCNV = 1e-3, toleranceMB = 1e-7)
% stop the Newton loop long before the discrete equations are actually solved,
% so its states carry a residual of their own. PyMRST drives to 1e-8, which
% makes it the *more* converged of the two - and the leftover difference then
% looks like a modelling disagreement when it is nothing but MRST's slack.
% Tightening MRST here removes that confound, so what the comparison measures is
% the physics rather than the two codes' stopping rules.
model.toleranceCNV = 1e-10;
model.toleranceMB  = 1e-13;
model = model.validateModel();

% -- a producer at the near end, drawing the reservoir down ------------------
%
% ONE perforation, deliberately. PyMRST's connection pressure drop is a single
% hydrostatic column per perforation - exact for a one-cell well, and an
% approximation for a multi-cell one (it does not do MRST's wellbore-topology
% sum). That approximation is a known, separate gap; with a three-cell well it
% contaminates this comparison at the 1e-3 level and would be mistaken for an
% aquifer error. A single perforation removes it, so what is left is the
% aquifer.
prodcells = find(G.cells.centroids(:, 1) < 500/nx + 1e-9 & ...
                 abs(G.cells.centroids(:, 3) - 2015) < 1e-9);
assert(numel(prodcells) == 1);
W = addWell([], G, rock, prodcells, 'Type', 'bhp', 'Val', 150*barsa, ...
            'Comp_i', [0, 1], 'sign', -1, 'Radius', 0.1, 'Name', 'PROD');

nstep = 20;
dt = repmat(20*day, nstep, 1);
schedule = simpleSchedule(dt, 'W', W);

% -- initial state -----------------------------------------------------------
state0 = initResSol(G, pRef, [0.2, 0.8]);
state0 = model.validateState(state0);        % seeds state0.aquiferSol

[wellSols, states, report] = simulateScheduleAD(state0, model, schedule);

nmini = zeros(1, numel(report.ControlstepReports));
for k = 1:numel(report.ControlstepReports)
    nmini(k) = numel(report.ControlstepReports{k}.StepReports);
end
assert(all(nmini == 1), ...
    'MRST sub-divided a report step; the comparison would be meaningless.');

p_hist  = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sw_hist = cell2mat(cellfun(@(s) s.s(:,1)',   states, 'UniformOutput', false));
aq_p    = cell2mat(cellfun(@(s) s.aquiferSol.pressure, states, 'UniformOutput', false));
aq_V    = cell2mat(cellfun(@(s) s.aquiferSol.volume,   states, 'UniformOutput', false));
aq_q    = cell2mat(cellfun(@(s) s.aquiferSol.flux(:)', states, 'UniformOutput', false));

qWs = cell2mat(cellfun(@(w) [w.qWs], wellSols, 'UniformOutput', false));
qOs = cell2mat(cellfun(@(w) [w.qOs], wellSols, 'UniformOutput', false));
bhp = cell2mat(cellfun(@(w) [w.bhp], wellSols, 'UniformOutput', false));

WI = W.WI;

save(fullfile(thisDir, 'ref_p48b_aquifer.mat'), '-v7', ...
     'p_hist', 'sw_hist', 'aq_p', 'aq_V', 'aq_q', ...
     'qWs', 'qOs', 'bhp', 'WI', ...
     'conncells', 'prodcells', 'Jval', 'Caq', 'alpha', 'depthconn', ...
     'depthaq', 'nx', 'nz', 'nstep', 'pRef');

fprintf('Wrote ref_p48b_aquifer.mat to %s\n', thisDir);
fprintf('  ministeps per report step: all %d\n', max(nmini));
fprintf('  aquifer p : %.3f -> %.3f bar\n', 250, aq_p(end)/barsa);
fprintf('  aquifer V : %.4e -> %.4e m3\n', 5.0e7, aq_V(end));
fprintf('  reservoir p (far cell): %.3f -> %.3f bar\n', ...
        pRef/barsa, p_hist(end, conncells(1))/barsa);
