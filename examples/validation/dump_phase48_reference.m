%% Dump Phase 48 reference: Fetkovich analytic aquifers, from MRST ad-core.
%
% Run in MATLAB after starting MRST. Writes ref_p48_aquifer.mat.
%
% The aquifer physics is self-contained: given the water phase pressure and
% shrinkage factor at each connection cell, MRST's AquiferModel computes an
% influx and then, after a step, updates its own pressure and volume. This dump
% isolates exactly that - it does NOT run a coupled reservoir solve - so the
% comparison tests the ported *formulas* (the Fetkovich relaxation, the
% hydrostatic head, the after-convergence update) rather than a whole
% simulation whose timestep controller no other code could be expected to
% reproduce. It follows the project's "fix the inputs, compare the algorithm"
% contract: the water pressure and shrinkage MRST evaluated are saved alongside
% its fluxes, and PyMRST replays its own formulas on those same numbers.
%
% Two flux samples are taken, at dt = 0 (the relaxation coefficient is exactly
% 1) and at a large dt (the coefficient is meaningfully below 1), so the
% Fetkovich time-constant term is actually exercised. Then the aquifer is drawn
% down over several steps against a *fixed* reservoir state, giving a
% deterministic pressure/volume trajectory.

mrstModule add ad-core ad-props ad-blackoil

gravity reset on;                 % gravity(3) ~ 9.80665, so the head term fires
thisDir = fileparts(mfilename('fullpath'));

% -- a small oil-water reservoir ---------------------------------------------
pRef = 200*barsa;
G = cartGrid([3, 1, 2], [300, 100, 20]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + 2000;   % put it at depth
G = computeGeometry(G);
nc = G.cells.num;                                     % 6
rock = makeRock(G, 100*milli*darcy, 0.25);

% The fluid details do not have to be matched by PyMRST: the water pressure and
% shrinkage MRST derives from this fluid are saved and fed straight into the
% PyMRST formulas, so any fluid with a water phase would do.
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',  [0.5, 2.0]*centi*poise, ...
                           'rho', [1000, 800], ...
                           'n',   [2, 2], ...
                           'c',   [4.0e-5, 1.0e-4]/barsa, ...
                           'pRef', pRef);

model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                             'gas', false);
model = model.validateModel();

% -- one aquifer, two connections on the top-layer boundary cells ------------
aquind.aquid     = 1;
aquind.conn      = 2;
aquind.pvttbl    = 3;
aquind.J         = 4;
aquind.C         = 5;
aquind.alpha     = 6;
aquind.depthconn = 7;
aquind.depthaq   = 8;

conncells = [1; 3];
nconn     = numel(conncells);
Jval      = [2.0e-11; 3.0e-11];    % connection productivity indices
Caq       = 1.0e-5/barsa;          % aquifer compressibility (per aquifer)
alpha     = [0.4; 0.6];            % influx shares (already sum to one)
depthaq   = 2010;                  % aquifer datum depth
depthconn = G.cells.centroids(conncells, 3);

aquifers = nan(nconn, 8);
aquifers(:, aquind.aquid)     = 1;
aquifers(:, aquind.conn)      = conncells;
aquifers(:, aquind.pvttbl)    = 1;
aquifers(:, aquind.J)         = Jval;
aquifers(:, aquind.C)         = Caq;         % per-connection (its aquifer's C)
aquifers(:, aquind.alpha)     = alpha;
aquifers(:, aquind.depthconn) = depthconn;
aquifers(:, aquind.depthaq)   = depthaq;

aquiferprops.C       = Caq;                  % per aquifer (used by the update)
aquiferprops.J       = sum(Jval);
aquiferprops.depthaq = depthaq;
aquiferprops.pvttbl  = 1;

initvals.pressures = 260*barsa;              % aquifer pressure (naq = 1)
initvals.volumes   = 1.0e8;                  % aquifer volume

aqmodel = AquiferModel(model, aquifers, aquind, aquiferprops, initvals);

% -- a fixed reservoir state (no solve; we only probe the aquifer) -----------
state = struct();
state.pressure = linspace(190, 210, nc)'*barsa;
state.s = [repmat(0.4, nc, 1), repmat(0.6, nc, 1)];
state = aqmodel.validateState(state);        % sets state.aquiferSol from initvals

% -- the water pressure and shrinkage MRST sees at the connections -----------
pp  = model.getProps(state, 'PhasePressures');
bb  = model.getProps(state, 'ShrinkageFactors');
ph  = model.getPhaseNames();
isw = (ph == 'W');
pW  = pp{isw}(conncells);
bW  = bb{isw}(conncells);

gv    = model.gravity;
gz    = gv(3);
rhoWS = model.fluid.rhoWS;

% -- flux samples: dt = 0 (coef == 1) and a large dt (coef < 1) --------------
DT_BIG = 2000*day;
q_dt0   = computeAquiferFluxes(aqmodel, state, 0);
q_dtbig = computeAquiferFluxes(aqmodel, state, DT_BIG);

% -- draw the aquifer down over several steps, reservoir state held fixed -----
dtU   = 30*day;
nstep = 10;
p_traj = zeros(nstep, 1);
v_traj = zeros(nstep, 1);
q_traj = zeros(nstep, nconn);

st = state;
for k = 1:nstep
    q = computeAquiferFluxes(aqmodel, st, dtU);
    q_traj(k, :) = q(:)';
    st = aqmodel.updateAfterConvergence(st, st, dtU, []);
    p_traj(k) = st.aquiferSol.pressure;
    v_traj(k) = st.aquiferSol.volume;
end

% -- everything PyMRST needs to replay the formulas --------------------------
aquid      = aquifers(:, aquind.aquid);
J          = Jval;
C_conn     = aquifers(:, aquind.C);
C_aq       = Caq;
p_aq0      = initvals.pressures;
V_aq0      = initvals.volumes;

save(fullfile(thisDir, 'ref_p48_aquifer.mat'), '-v7', ...
     'conncells', 'aquid', 'J', 'C_conn', 'C_aq', 'alpha', ...
     'depthconn', 'depthaq', 'p_aq0', 'V_aq0', ...
     'pW', 'bW', 'rhoWS', 'gz', ...
     'q_dt0', 'q_dtbig', 'DT_BIG', ...
     'q_traj', 'p_traj', 'v_traj', 'dtU', 'nstep');

fprintf('Wrote ref_p48_aquifer.mat to %s\n', thisDir);
fprintf('  q(dt=0)   = %s\n', mat2str(q_dt0(:)', 6));
fprintf('  q(dt=big) = %s\n', mat2str(q_dtbig(:)', 6));
fprintf('  aquifer p: %.4f -> %.4f bar over %d steps\n', ...
        p_aq0/barsa, p_traj(end)/barsa, nstep);
