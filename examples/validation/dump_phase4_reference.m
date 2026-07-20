%% Dump Phase 4 reference: compressible single-phase AD simulator.
%
% Mirror of examples/single_phase_compressible_ad.py, which itself mirrors
% MRST's flowSolverTutorialAD.m. The hydrostatic initial state uses the
% closed-form solution instead of ode23 so both languages start identically.

thisDir = fileparts(mfilename('fullpath'));

%% Model setup (flowSolverTutorialAD.m parameters)
G    = computeGeometry(cartGrid([10, 10, 10], [200, 200, 50]));
rock = makeRock(G, 30 * milli * darcy, 0.3);

cr   = 1e-6 / barsa;   p_r  = 200 * barsa;
pv_r = poreVolume(G, rock);
pv   = @(p) pv_r .* exp(cr * (p - p_r));

mu    = 5 * centi * poise;
c     = 1e-3 / barsa;
rho_r = 850;   rhoS = 750;
rho   = @(p) rho_r .* exp(c * (p - p_r));

nperf   = 8;
I = repmat(2, [nperf, 1]); J = (1 : nperf).' + 1; K = repmat(5, [nperf, 1]);
cellInx = sub2ind(G.cartDims, I, J, K);
gravity reset on, g = norm(gravity);
W  = addWell([], G, rock, cellInx, 'Name', 'P1', 'Dir', 'y');

% Closed-form vertical equilibrium: dp/dz = g rho(p), p(0) = p_r.
a = c * g * rho_r * exp(-c * p_r);
z = G.cells.centroids(:, 3);
p_init = -log(exp(-c * p_r) - a * z) / c;

%% Discrete operators
N  = double(G.faces.neighbors);
intInx = all(N ~= 0, 2);
N  = N(intInx, :);
hT = computeTrans(G, rock);
cf = G.cells.faces(:, 1);
T  = 1 ./ accumarray(cf, 1 ./ hT, [G.faces.num, 1]);
T  = T(intInx);
n  = size(N, 1);
C  = sparse(repmat((1 : n).', [2, 1]), N, repmat([-1, 1], [n, 1]), ...
            n, G.cells.num);
grad = @(x) C * x;
div  = @(x) -C' * x;
avg  = @(x) 0.5 * (x(N(:, 1)) + x(N(:, 2)));

gradz  = grad(G.cells.centroids(:, 3));
flux   = @(p) -(T / mu) .* (grad(p) - g * avg(rho(p)) .* gradz);
presEq = @(p, p0, dt) (1 / dt) * (pv(p) .* rho(p) - pv(p0) .* rho(p0)) ...
                      + div(avg(rho(p)) .* flux(p));

wc = W(1).cells;  WI = W(1).WI;  dZ = W(1).dZ;
p_conn = @(bhp)    bhp + g * dZ .* rho(bhp);
q_conn = @(p, bhp) WI .* (rho(p(wc)) / mu) .* (p_conn(bhp) - p(wc));
rateEq = @(p, bhp, qS) qS - sum(q_conn(p, bhp)) / rhoS;
ctrlEq = @(bhp)        bhp - 100 * barsa;

%% Time loop
[p_ad, bhp_ad, qS_ad] = initVariablesADI(p_init, p_init(wc(1)), 0);
nc = G.cells.num;
[pIx, bhpIx, qSIx] = deal(1 : nc, nc + 1, nc + 2);

numSteps = 52;  totTime = 365 * day;  dt = totTime / numSteps;
tol = 1e-5;  maxits = 10;

p_hist   = zeros(numSteps + 1, nc);  p_hist(1, :) = value(p_ad);
bhp_hist = zeros(numSteps + 1, 1);   bhp_hist(1) = value(bhp_ad);
qs_hist  = zeros(numSteps + 1, 1);
nits     = zeros(numSteps, 1);

for step = 1 : numSteps
   p0 = value(p_ad);
   converged = false;  nit = 0;
   while ~converged && (nit < maxits)
      eq1     = presEq(p_ad, p0, dt);
      eq1(wc) = eq1(wc) - q_conn(p_ad, bhp_ad);
      eqs = {eq1, rateEq(p_ad, bhp_ad, qS_ad), ctrlEq(bhp_ad)};
      eq  = cat(eqs{:});
      res = eq.val;
      upd = -(eq.jac{1} \ res);
      p_ad.val   = p_ad.val   + upd(pIx);
      bhp_ad.val = bhp_ad.val + upd(bhpIx);
      qS_ad.val  = qS_ad.val  + upd(qSIx);
      converged = ~(norm(res) > tol);
      nit = nit + 1;
   end
   assert(converged, 'Newton did not converge in step %d', step);
   nits(step)          = nit;
   p_hist(step + 1, :) = value(p_ad);
   bhp_hist(step + 1)  = value(bhp_ad);
   qs_hist(step + 1)   = value(qS_ad);
end

gravity reset; gravity off
save(fullfile(thisDir, 'ref_p4_ad.mat'), '-v7', ...
     'p_hist', 'bhp_hist', 'qs_hist', 'p_init', 'WI', 'dZ', 'nits');
fprintf('Wrote Phase-4 reference to %s\n', thisDir);
