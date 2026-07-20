%% Dump Phase 47 reference: vaporised oil (rv), from MRST ad-blackoil.
%
% Run in MATLAB after starting MRST. Writes ref_p47_vapoil.mat.
%
% A gas-condensate column: a gas cap over an oil leg, produced from the middle.
% It is built to exercise every branch of the black-oil variable switch at once:
%
%   - the gas-cap cells start with **no oil phase at all** (status 2), so `rv`
%     is their primary variable - the state PyMRST has never had;
%   - the oil-leg cells start **undersaturated** (status 1), so `rs` is theirs;
%   - and as the well draws the pressure down, gas comes out of solution in the
%     oil leg and oil condenses out of the gas cap, so cells cross into the
%     saturated state (status 3) in both directions.
%
% The PVT is analytic rather than tabulated, on purpose: both codes then
% evaluate the *same closed-form functions*, and any difference is the physics
% and not the interpolation. bO and muO depend on (p, rs, saturated-flag) and
% bG and muG on (p, rv, saturated-flag), so the flags - which are different
% flags, and easy to swap - are exercised too.

mrstModule add ad-core ad-props ad-blackoil

% Gravity off, deliberately. With it on, a uniform initial pressure over a
% 100 m column is nowhere near equilibrium, and the first timestep segregates
% the whole thing into saturated three-phase cells - which erases the status-1
% and status-2 states this case exists to test before they have been used. With
% it off, the only thing that changes a cell's status is the *flash*, which is
% exactly what is being validated.
gravity reset off;
thisDir = fileparts(mfilename('fullpath'));

nz    = 20;
top   = 2000;
thick = 100;
pRef  = 300*barsa;

G = cartGrid([1, 1, nz], [10, 10, thick]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + top;
G = computeGeometry(G);
nc = G.cells.num;
% 1 mD, not 100. At 100 mD this column drains to the bottom-hole pressure in
% *minutes*, so a 15-day step resolves no transient at all - every cell simply
% sits at the well's bhp and the comparison tests nothing but the equilibrium.
% At 1 mD the pressure front actually travels, and the saturations, rs and rv
% have a history worth comparing.
rock = makeRock(G, 1*milli*darcy, 0.25);

% -- the analytic PVT ---------------------------------------------------------
rhoWS = 1000; rhoOS = 800; rhoGS = 1.2;
cW = 4.0e-5/barsa;  muW0 = 0.4*centi*poise;
cO = 1.2e-4/barsa;  muO0 = 2.0*centi*poise;
cG = 3.0e-3/barsa;  muG0 = 0.02*centi*poise;
Rs0 = 150;          % sm3 gas / sm3 oil at pRef
Rv0 = 8.0e-4;       % sm3 oil / sm3 gas at pRef  (Rs0*Rv0 = 0.12 << 1)
gamO = 1.0e-3;      % how much dissolved gas swells the oil
gamG = 50;          % how much vaporised oil densifies the gas
dO   = 2.0e-3;      % dissolved gas thins the oil
dG   = 100;         % vaporised oil thickens the gas

rsSat = @(p) Rs0 .* (p ./ pRef);
rvSat = @(p) Rv0 .* (p ./ pRef);

fluid = struct();
fluid.rhoWS = rhoWS;  fluid.rhoOS = rhoOS;  fluid.rhoGS = rhoGS;
% injected fluid comes in at the same surface densities
fluid.rhoSinj = [rhoWS, rhoOS, rhoGS];
fluid.pvMultR = @(p) 1 + 1.0e-5/barsa .* (p - pRef);

fluid.bW  = @(p) 1 + cW.*(p - pRef);
fluid.muW = @(p) muW0 + 0.*p;

fluid.rsSat = rsSat;
fluid.rvSat = rvSat;

% flag == true  -> the phase is saturated, so use the saturation curve
fluid.bO  = @(p, rs, flag) (1 + cO.*(p - pRef)) ./ ...
                           (1 + gamO.*(flag.*rsSat(p) + ~flag.*rs));
fluid.muO = @(p, rs, flag) muO0 .* (1 + 5.0e-4/barsa.*(p - pRef)) ./ ...
                           (1 + dO.*(flag.*rsSat(p) + ~flag.*rs));
fluid.bG  = @(p, rv, flag) (1 + cG.*(p - pRef)) .* ...
                           (1 + gamG.*(flag.*rvSat(p) + ~flag.*rv));
fluid.muG = @(p, rv, flag) muG0 .* (1 + 2.0e-4/barsa.*(p - pRef)) .* ...
                           (1 + dG.*(flag.*rvSat(p) + ~flag.*rv));

swcon = 0.1;
fluid.sWcon = swcon;
fluid.krW  = @(sw) max((sw - swcon)./(1 - swcon), 0).^2;
fluid.krG  = @(sg) max(sg, 0).^2;
fluid.krOW = @(so) max(so, 0).^2;
fluid.krOG = @(so) max(so, 0).^2;

model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true, ...
                                'vapoil', true);
model = model.validateModel();

% -- initial state: a gas cap (no oil at all) over an undersaturated oil leg ---
ngc = 6;                                   % cells in the gas cap
z   = G.cells.centroids(:, 3);
p0  = pRef + 0.*z;

sw = repmat(swcon, nc, 1);
so = zeros(nc, 1);
sg = zeros(nc, 1);
cap = (1:nc)' <= ngc;
sg(cap)  = 1 - swcon;                      % gas cap: NO oil phase -> status 2
so(~cap) = 1 - swcon;                      % oil leg: NO free gas  -> status 1

rs = zeros(nc, 1);
rv = zeros(nc, 1);
rs(~cap) = 0.55 * rsSat(p0(~cap));         % well below the bubble point
rv(cap)  = 0.55 * rvSat(p0(cap));          % well below the dew point

state0 = struct('pressure', p0, 's', [sw, so, sg], 'rs', rs, 'rv', rv);

% -- two cases, because one is not enough ------------------------------------
%
%   A. "steady": a gentle 230 bar producer. Every cell stays where it started -
%      the gas cap remains **status 2**, with `rv` as its primary variable, and
%      the oil leg **status 1**, with `rs`. This is the steady exercise of both
%      undersaturated states, and status 2 is one this port never had.
%
%   B. "depletion": a 160 bar producer on shorter steps. Cells cross into
%      **status 3** as gas breaks out of the oil - so the *transitions* are
%      exercised, which is where a variable switch actually goes wrong.
%
% A model that got the switch right in one and wrong in the other would pass
% either test on its own.
%
% Both cases are chosen so MRST takes **exactly one ministep per report step**.
% If it had to sub-divide, it would be solving a different discrete problem
% from a single-step solve, and no other code could be expected to reproduce
% it; the ministep counts are printed so that assumption is checked and not
% merely hoped for.

cases = struct('tag',   {'steady',  'depletion'}, ...
               'bhp',   {230*barsa, 160*barsa}, ...
               'dt',    {15*day,    7.5*day}, ...
               'nstep', {16,        32});

for ci = 1:numel(cases)
    W = addWell([], G, rock, floor(nz/2), 'Type', 'bhp', ...
                'Val', cases(ci).bhp, 'Comp_i', [0, 1, 0], 'sign', -1, ...
                'Radius', 0.1, 'Name', 'PROD');
    dt = repmat(cases(ci).dt, cases(ci).nstep, 1);
    schedule = simpleSchedule(dt, 'W', W);
    [wellSols, states, report] = simulateScheduleAD(state0, model, schedule);

    nmini = zeros(1, numel(report.ControlstepReports));
    for k = 1:numel(report.ControlstepReports)
        nmini(k) = numel(report.ControlstepReports{k}.StepReports);
    end
    assert(all(nmini == 1), ...
        'MRST sub-divided a report step; the comparison would be meaningless.');

    t = cases(ci).tag;
    out.([t '_p'])   = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
    out.([t '_sw'])  = cell2mat(cellfun(@(s) s.s(:,1)',   states, 'UniformOutput', false));
    out.([t '_so'])  = cell2mat(cellfun(@(s) s.s(:,2)',   states, 'UniformOutput', false));
    out.([t '_sg'])  = cell2mat(cellfun(@(s) s.s(:,3)',   states, 'UniformOutput', false));
    out.([t '_rs'])  = cell2mat(cellfun(@(s) s.rs',       states, 'UniformOutput', false));
    out.([t '_rv'])  = cell2mat(cellfun(@(s) s.rv',       states, 'UniformOutput', false));
    out.([t '_st'])  = cell2mat(cellfun(@(s) double(s.status)', states, 'UniformOutput', false));
    out.([t '_qOs']) = cell2mat(cellfun(@(w) [w.qOs], wellSols, 'UniformOutput', false));
    out.([t '_qGs']) = cell2mat(cellfun(@(w) [w.qGs], wellSols, 'UniformOutput', false));
    out.([t '_qWs']) = cell2mat(cellfun(@(w) [w.qWs], wellSols, 'UniformOutput', false));
    out.([t '_bhp']) = cell2mat(cellfun(@(w) [w.bhp], wellSols, 'UniformOutput', false));
    out.([t '_WI'])  = W.WI;
    out.([t '_dt'])    = cases(ci).dt;
    out.([t '_nstep']) = cases(ci).nstep;

    fprintf('  %-10s: %d ministeps max; status end %s\n', t, max(nmini), ...
            mat2str(out.([t '_st'])(end, :)));
end

steady_p = out.steady_p;  steady_sw = out.steady_sw;  steady_so = out.steady_so;
steady_sg = out.steady_sg; steady_rs = out.steady_rs; steady_rv = out.steady_rv;
steady_st = out.steady_st; steady_qOs = out.steady_qOs; steady_qGs = out.steady_qGs;
steady_qWs = out.steady_qWs; steady_bhp = out.steady_bhp; steady_WI = out.steady_WI;
steady_dt = out.steady_dt; steady_nstep = out.steady_nstep;
depletion_p = out.depletion_p;  depletion_sw = out.depletion_sw;
depletion_so = out.depletion_so; depletion_sg = out.depletion_sg;
depletion_rs = out.depletion_rs; depletion_rv = out.depletion_rv;
depletion_st = out.depletion_st; depletion_qOs = out.depletion_qOs;
depletion_qGs = out.depletion_qGs; depletion_qWs = out.depletion_qWs;
depletion_bhp = out.depletion_bhp; depletion_WI = out.depletion_WI;
depletion_dt = out.depletion_dt; depletion_nstep = out.depletion_nstep;

save(fullfile(thisDir, 'ref_p47_vapoil.mat'), '-v7', ...
     'steady_p', 'steady_sw', 'steady_so', 'steady_sg', 'steady_rs', ...
     'steady_rv', 'steady_st', 'steady_qOs', 'steady_qGs', 'steady_qWs', ...
     'steady_bhp', 'steady_WI', 'steady_dt', 'steady_nstep', ...
     'depletion_p', 'depletion_sw', 'depletion_so', 'depletion_sg', ...
     'depletion_rs', 'depletion_rv', 'depletion_st', 'depletion_qOs', ...
     'depletion_qGs', 'depletion_qWs', 'depletion_bhp', 'depletion_WI', ...
     'depletion_dt', 'depletion_nstep', ...
     'nz', 'top', 'thick', 'pRef', 'ngc', 'swcon');

fprintf('Wrote ref_p47_vapoil.mat to %s\n', thisDir);
