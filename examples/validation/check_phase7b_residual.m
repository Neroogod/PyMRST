%% Phase 7b validation: evaluate MRST's OWN MICP residual at PyMRST's states.
%
% This is the decisive check that the PyMRST port implements exactly the
% same discrete equations as ad-micp: we take the states PyMRST converged
% to (written by validate_phase7b.py as py_p7b_states.mat) and evaluate
% MRST's equationsMICP residual at them. A residual at round-off level
% proves the two implementations solve the identical system.
%
% Context: MRST's own solver does NOT drive these equations to zero. At
% its own reported states the microorganism residual is ~1e-8 and its own
% checkConvergence marks CNV_W / MB_W / microorganism / biofilm as failing.
% The ad-micp examples request nonlinearTolerance = toleranceMB = 1e-14,
% which the solver never attains, so it returns partially converged states.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-blackoil ad-core ad-micp
gravity reset; gravity off

PY = load(fullfile(thisDir, 'py_p7b_states.mat'));

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
fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS];
fluid.Comax = max(PY.Mmat(:, 5));
fluid.Cumax = max(PY.Mmat(:, 6));

G.injectionwellonboundary = 1;
G.cellsinjectionwell = 1;

f = boundaryFaces(G);
f = f(abs(G.faces.normals(f, 1)) > 0 & ...
      G.faces.centroids(f, 1) > X(end - 1));

model = MICPModel(G, rock, fluid);
model.toleranceMB = 1e-14;  model.nonlinearTolerance = 1e-14;
nc = G.cells.num;
M = PY.Mmat;

% A single MRST step gives a state container with a valid wellSol; we then
% overwrite its fields with the PyMRST values at each report step.
W0 = addWell([], G, rock, 1, 'Type', 'rate', 'Comp_i', [1, 0], ...
             'Val', M(1, 3), 'Radius', 0.15);
W0.m = M(1, 4);  W0.o = M(1, 5);  W0.u = M(1, 6);
bc0 = addBC([], f, 'pressure', atm, 'sat', [0 0]);
bc0.m = zeros(numel(f), 1); bc0.o = bc0.m; bc0.u = bc0.m;
bc0.b = bc0.m;              bc0.c = bc0.m;
init = initState(G, W0, atm, [1, 0]);
init.m = zeros(nc, 1); init.o = init.m; init.u = init.m;
init.b = init.m;       init.c = init.m;
[~, tmp] = simulateScheduleAD(init, model, ...
              simpleSchedule(PY.dts(1), 'W', W0, 'bc', bc0));
template = tmp{1};

worst = zeros(numel(PY.dts), 7);
for i = 1 : numel(PY.dts)
   k = PY.ctrl(i);
   W = addWell([], G, rock, 1, 'Type', 'rate', 'Comp_i', [1, 0], ...
               'Val', M(k, 3), 'Radius', 0.15);
   W.m = M(k, 4);  W.o = M(k, 5);  W.u = M(k, 6);
   bc = addBC([], f, 'pressure', atm, 'sat', [0 0]);
   bc.m = zeros(numel(f), 1); bc.o = bc.m; bc.u = bc.m;
   bc.b = bc.m;               bc.c = bc.m;
   forces = struct('W', W, 'bc', bc, 'src', []);
   mdl = model.validateModel(forces);

   if i == 1
      st0 = template;
      st0.pressure = init.pressure; st0.s = init.s;
      st0.m = init.m; st0.o = init.o; st0.u = init.u;
      st0.b = init.b; st0.c = init.c;
   else
      st0 = setPyState(template, PY, i - 1);
   end
   st = setPyState(template, PY, i);
   st.wellSol(1).bhp = PY.bhp(i);

   problem = mdl.getEquations(st0, st, PY.dts(i), forces, 'resOnly', true);
   for j = 1 : 7
      worst(i, j) = norm(value(problem.equations{j}), inf);
   end
end

names = {'water', 'oil', 'microorganism', 'oxygen', 'urea', ...
         'biofilm', 'calcite'};
fprintf('\nMRST residuals evaluated at PyMRST states (max over 39 steps):\n');
for j = 1 : 7
   fprintf('  %-14s %.3e\n', names{j}, max(worst(:, j)));
end
fprintf('\nOVERALL MAX = %.3e\n', max(worst(:)));
if max(worst(:)) < 1e-10
   fprintf('PASS: PyMRST states satisfy MRST''s own MICP equations.\n');
else
   fprintf('FAIL: residual too large.\n');
end

function st = setPyState(st, PY, i)
   st.pressure = PY.p_hist(i, :)';
   st.m = PY.m_hist(i, :)';
   st.o = PY.o_hist(i, :)';
   st.u = PY.u_hist(i, :)';
   st.b = PY.b_hist(i, :)';
   st.c = PY.c_hist(i, :)';
   st.s = [ones(numel(st.pressure), 1), zeros(numel(st.pressure), 1)];
end
