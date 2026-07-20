%% Dump Phase 11 reference: MRST flow diagnostics.
%
% A heterogeneous quarter five-spot: time-of-flight (forward and backward),
% well tracer partitions, and the F-Phi / Lorenz / sweep measures derived
% from them.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add diagnostics incomp

n = 32;
G = computeGeometry(cartGrid([n, n, 1], [500, 500, 10]));

% Deterministic heterogeneous permeability (same formula as the Python side)
kvals = (10 + mod(7 * (0 : G.cells.num - 1)', 100)) * milli * darcy;
rock  = makeRock(G, kvals, 0.2);
fluid = initSingleFluid('mu', 1 * centi * poise, 'rho', 1000);
hT    = computeTrans(G, rock);

W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', 1e-3, 'Radius', 0.1, ...
            'Comp_i', 1);
W = addWell(W, G, rock, G.cells.num, 'Type', 'bhp', ...
            'Val', 100 * barsa, 'Radius', 0.1, 'Comp_i', 1);

state = incompTPFA(initState(G, W, 0), G, hT, fluid, 'wells', W);

D = computeTOFandTracer(state, G, rock, 'wells', W);

tof_f  = D.tof(:, 1);
tof_b  = D.tof(:, 2);
tr_inj = D.itracer;
tr_prd = D.ptracer;

pv = poreVolume(G, rock);
[F, Phi] = computeFandPhi(pv, D.tof);
Lc = computeLorenz(F, Phi);
[Ev, tD] = computeSweep(F, Phi);

p    = state.pressure;
flux = state.flux;

save(fullfile(thisDir, 'ref_p11_diagnostics.mat'), '-v7', ...
     'tof_f', 'tof_b', 'tr_inj', 'tr_prd', 'F', 'Phi', 'Lc', ...
     'Ev', 'tD', 'p', 'flux', 'kvals');
fprintf('Wrote Phase-11 diagnostics reference (Lorenz = %.4f)\n', Lc);
