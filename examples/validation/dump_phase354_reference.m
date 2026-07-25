%% Dump Phase 354 reference: live-oil equilibration density, from MRST ad-core.
%
% Run in MATLAB after starting MRST. Writes ref_p354_equil_disgas.mat.
%
% Phase 42 validated equilibration with an *analytic, dead-oil* fluid. This does
% the case that fluid could never reach: a **live-oil** column, where the oil
% shrinkage factor depends on the dissolved gas `rs`, and the density that
% carries the column carries its gas with it. That path was silently broken in
% PyMRST (a `None` rs produced NaN and hung the ODE integrator) until it was
% given MRST's own density formula - so this pins that formula against MRST.
%
% The discriminating choice is a **constant** `rs = rsSat(p_datum)`. Because the
% saturated `rsSat` rises with pressure, a constant rs is:
%   - >= rsSat *above* the datum (lower p): the oil is saturated, rs capped;
%   - <  rsSat *below* the datum (higher p): the oil is undersaturated.
% So one column exercises *both* branches of MRST's `getOilDensity`
% (`rs = min(rs, rsSat)`, saturated iff `rs >= rsSat`), which a fully-saturated
% column would not.
%
% The fluid is SPE 1's real PVTO/PVDG tables; only the initialization is
% replaced by an EQUIL-style hydrostatic region.

mrstModule add ad-core ad-props ad-blackoil deckformat

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

% -- SPE 1's fluid, on a tall single-column grid ------------------------------
deck = convertDeckUnits(readEclipseDeck( ...
    fullfile(thisDir, '..', '..', '..', 'mrst-2024a-Geomech', ...
             'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA')));
fluid = initDeckADIFluid(deck);

nz = 20;
top = 2400; thick = 200;                    % 2400 - 2600 m
G = cartGrid([1, 1, nz], [10, 10, thick]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + top;
G = computeGeometry(G);
rock = makeRock(G, 100*milli*darcy, 0.25);

model = GenericBlackOilModel(G, rock, fluid, 'water', true, 'oil', true, ...
                             'gas', true, 'disgas', true);
model = model.validateModel();

% -- an equilibration region with a constant rs ------------------------------
datum_depth    = 2500;
datum_pressure = 250*barsa;
woc = 2580;                                 % water-oil contact
goc = 2420;                                 % gas-oil contact
contacts = [goc, woc];                       % shallowest-first, as MRST wants

rsSatF = fluid.rsSat;                        % may be a per-region cell array
if iscell(rsSatF); rsSatF = rsSatF{1}; end
rs_const = rsSatF(datum_pressure);           % saturated at the datum

% MRST's initStateBlackOilAD expects region.rs to be a *function handle*
% rs(p, z) (RSVD-style), even for a constant. Pass a constant-valued handle.
rs_fn = @(p, z) rs_const + 0*p;

region = getInitializationRegionsBlackOil(model, contacts, ...
    'datum_depth', datum_depth, 'datum_pressure', datum_pressure, ...
    'rs', rs_fn);

[state, pressures] = initStateBlackOilAD(model, region);

z = G.cells.centroids(:, 3);

% the discriminator: which cells came out saturated vs undersaturated
rsSat_cell = rsSatF(state.pressure);         %#ok (rsSatF unwrapped above)
saturated = state.rs >= rsSat_cell - 1e-6*max(rsSat_cell);

save(fullfile(thisDir, 'ref_p354_equil_disgas.mat'), '-v7', ...
     'z', 'nz', 'top', 'thick', 'datum_depth', 'datum_pressure', ...
     'woc', 'goc', 'rs_const', ...
     'pressures', 'saturated', ...
     'state');   % state.pressure, state.s (n x 3), state.rs

fprintf('Wrote ref_p354_equil_disgas.mat to %s\n', thisDir);
fprintf('  rs (constant)     : %.4f\n', rs_const);
fprintf('  p range (bar)     : %.3f - %.3f\n', ...
        min(state.pressure)/barsa, max(state.pressure)/barsa);
fprintf('  rs range          : %.4f - %.4f\n', min(state.rs), max(state.rs));
fprintf('  saturated cells   : %d of %d\n', nnz(saturated), nz);
