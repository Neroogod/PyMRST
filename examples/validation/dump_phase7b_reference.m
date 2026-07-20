%% Dump Phase 7b reference: ad-micp 1D case (runMICP1DCase, truncated).
%
% Same model/schedule as the module's runMICP1DCase example, but stopped
% after the microbe + oxygen injection phases so the run is short and the
% states are fully defined for comparison with PyMRST's MICPModelAD.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-blackoil ad-core ad-micp
gravity reset; gravity off

L = 75; l = 25; dw = 0.5; dl = 0.25;
X = [0 dw dw + dl : dl : l  L * exp(-1.075 : 0.025 : 0)];
G = computeGeometry(tensorGrid(X, [0 1], [0 1]));

C = ones(G.cells.num, 1);
K0 = 1e-12 * C;  porosity = 0.2;
rock = makeRock(G, K0, porosity);

fluid.muw = 2.535e-4;  fluid.bW = @(p) 0 * p + 1;  fluid.rhoWS = 1045;
fluid.rho_b = 35;      fluid.rho_c = 2710;    fluid.k_str = 2.6e-10;
fluid.diffm = 2.1e-9;  fluid.diffo = 2.32e-9; fluid.diffu = 1.38e-9;
fluid.alphaL = 1e-3;   fluid.alphaT = 4e-4;   fluid.eta = 3;
fluid.k_o = 2e-5;      fluid.k_u = 21.3;      fluid.mu = 4.17e-5;
fluid.mu_u = 0.0161;   fluid.k_a = 8.51e-7;   fluid.k_d = 3.18e-7;
fluid.Y = 0.5;         fluid.Yuc = 1.67;      fluid.F = 0.5;
fluid.crit = 0.1;      fluid.kmin = 1e-20;    fluid.cells = C;
fluid.ptol = 1e-4;
fluid.K = @(poro) (K0 .* ((poro - fluid.crit) / (porosity - fluid.crit)) ...
          .^ fluid.eta + fluid.kmin) .* K0 ./ (K0 + fluid.kmin) .* ...
          (poro > fluid.crit) + fluid.kmin .* (poro <= fluid.crit);
fluid.bO = fluid.bW;   fluid.rhoOS = fluid.rhoWS;
% This MRST install carries a local "OMO modification" in SimpleWell.m that
% reads fluid.rhoSinj for injectors (injected-fluid surface density, used by
% the CO2 leakage scripts). Stock ad-micp does not define it, so the example
% cannot run without this line. Setting it to the reservoir densities
% reproduces stock behavior; with gravity off and one perforation, cdp = 0
% either way, so it has no effect on this case.
fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS];

% Injection strategy: microbes, chase water, rest, oxygen, chase water.
N = 5;
M = zeros(N, 6);
dt_on = 20 * minute;  dt_off = hour;
M(1, :) = [4 * hour,  dt_on,  2 / day, 0.01, 0,    0];
M(2, :) = [2 * hour,  dt_on,  2 / day, 0,    0,    0];
M(3, :) = [3 * hour,  dt_off, 0,       0,    0,    0];
M(4, :) = [4 * hour,  dt_on,  2 / day, 0,    0.04, 0];
M(5, :) = [2 * hour,  dt_on,  2 / day, 0,    0,    0];

r = 0.15;
W = addWell([], G, rock, 1, 'Type', 'rate', 'Comp_i', [1, 0], ...
            'Val', M(1, 3), 'Radius', r);
W.m = M(1, 4);  W.o = M(1, 5);  W.u = M(1, 6);

G.injectionwellonboundary = 1;
G.cellsinjectionwell = 1;

f = boundaryFaces(G);
f = f(abs(G.faces.normals(f, 1)) > 0 & ...
      G.faces.centroids(f, 1) > X(end - 1));
bc = addBC([], f, 'pressure', atm, 'sat', [0 0]);
bc.m = zeros(size(bc.sat, 1), 1);  bc.o = bc.m;  bc.u = bc.m;
bc.b = bc.m;                       bc.c = bc.m;

nt = sum(M(:, 1) ./ M(:, 2));
schedule = simpleSchedule(repmat(dt_on, nt, 1), 'W', W, 'bc', bc);
for i = 2 : N
   schedule.control(i) = schedule.control(i - 1);
   k = sum(M(1 : i - 1, 1) ./ M(1 : i - 1, 2)) + 1;
   schedule.step.control(k : end) = i;
   schedule.step.val(k : end) = M(i, 2);
   schedule.control(i).W.val = M(i, 3);
   schedule.control(i).W.m = M(i, 4);
   schedule.control(i).W.o = M(i, 5);
   schedule.control(i).W.u = M(i, 6);
end

fluid.Comax = max(M(:, 5));
fluid.Cumax = max(M(:, 6));

model = MICPModel(G, rock, fluid);
model.toleranceMB = 1e-14;
model.nonlinearTolerance = 1e-14;
% NOTE: the ad-micp examples set toleranceMB/nonlinearTolerance but leave
% toleranceCNV at its 1e-3 default, and MICPModel uses CNV convergence.
% That stops Newton early (residuals up to ~1e-3 * pv/dt survive), which
% shifts the microbe field by ~2%. Tighten it so this reference is a
% properly converged solution of the discrete system.
model.toleranceCNV = 1e-12;
solver = getNonLinearSolver(model);
solver.LinearSolver.tolerance = 1e-14;
solver.maxIterations = 50;

state0 = initState(G, W, atm, [1, 0]);
state0.m = zeros(G.cells.num, 1);  state0.o = state0.m;
state0.u = state0.m;               state0.b = state0.m;
state0.c = state0.m;

[~, states] = simulateScheduleAD(state0, model, schedule, ...
                                 'NonLinearSolver', solver);

ns = numel(states);  nc = G.cells.num;
p_hist = zeros(ns, nc);  m_hist = zeros(ns, nc);  o_hist = zeros(ns, nc);
u_hist = zeros(ns, nc);  b_hist = zeros(ns, nc);  c_hist = zeros(ns, nc);
for i = 1 : ns
   p_hist(i, :) = states{i}.pressure;
   m_hist(i, :) = states{i}.m;
   o_hist(i, :) = states{i}.o;
   u_hist(i, :) = states{i}.u;
   b_hist(i, :) = states{i}.b;
   c_hist(i, :) = states{i}.c;
end
dts = schedule.step.val;
ctrl = schedule.step.control;
Mmat = M;
WI = W.WI;
xcoords = G.cells.centroids(:, 1);

save(fullfile(thisDir, 'ref_p7b_micp1d.mat'), '-v7', ...
     'p_hist', 'm_hist', 'o_hist', 'u_hist', 'b_hist', 'c_hist', ...
     'dts', 'ctrl', 'Mmat', 'WI', 'xcoords');
fprintf('Wrote Phase-7b MICP reference (%d steps) to %s\n', ns, thisDir);
