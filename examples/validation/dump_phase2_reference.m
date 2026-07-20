%% Dump Phase 2 reference solutions from MATLAB MRST for PyMRST validation.
%
% Run in MATLAB after MRST startup. Writes one .mat per case into this
% folder; validate_phase2.py then compares PyMRST solutions against them.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add incomp
gravity reset; gravity off

%% Case 1: quarter five-spot with sources (32x32, heterogeneous perm)
G = computeGeometry(cartGrid([32, 32, 1], [500, 500, 10]));
% Deterministic heterogeneous permeability, same formula as Python side.
kvals = (10 + mod(7 * (0 : G.cells.num - 1)', 100)) * milli * darcy;
rock  = makeRock(G, kvals, 0.2);
fluid = initSingleFluid('mu', 1 * centi * poise, 'rho', 1000);
hT    = computeTrans(G, rock);
rate  = 1e-3;
src   = addSource([] , 1          ,  rate);
src   = addSource(src, G.cells.num, -rate);
state = incompTPFA(initResSol(G, 0), G, hT, fluid, 'src', src);
p = state.pressure; flux = state.flux; fp = state.facePressure;
save(fullfile(thisDir, 'ref_p2_qfs.mat'), '-v7', 'p', 'flux', 'fp', 'hT');

%% Case 2: Dirichlet + Neumann boundary conditions (20x10x5)
G = computeGeometry(cartGrid([20, 10, 5], [200, 100, 50]));
kvals = (1 + mod(3 * (0 : G.cells.num - 1)', 50)) * milli * darcy;
rock  = makeRock(G, kvals, 0.25);
hT    = computeTrans(G, rock);
bd    = any(G.faces.neighbors == 0, 2);
west  = find(bd & G.faces.centroids(:, 1) < 1e-10);
east  = find(bd & G.faces.centroids(:, 1) > 200 - 1e-10);
south = find(bd & G.faces.centroids(:, 2) < 1e-10);
bc    = addBC([] , west , 'pressure', 200 * barsa);
bc    = addBC(bc , east , 'pressure', 100 * barsa);
bc    = addBC(bc , south, 'flux'    , 1e-6);
state = incompTPFA(initResSol(G, 0), G, hT, fluid, 'bc', bc);
p = state.pressure; flux = state.flux; fp = state.facePressure;
save(fullfile(thisDir, 'ref_p2_bc.mat'), '-v7', 'p', 'flux', 'fp', 'hT');

%% Case 3: rate injector + bhp producer (10x10x3, gravity on)
gravity reset; gravity on
G = computeGeometry(cartGrid([10, 10, 3], [1000, 1000, 30]));
kvals = repmat([100, 100, 10] * milli * darcy, G.cells.num, 1);
rock  = makeRock(G, kvals, 0.3);
hT    = computeTrans(G, rock);
W  = addWell([], G, rock, 1 : 100 : 201, ...       % vertical injector, i=j=1
             'Type', 'rate', 'Val', 5e-3, 'Radius', 0.1, 'Comp_i', 1);
W  = addWell(W, G, rock, 100 : 100 : 300, ...      % vertical producer, i=j=10
             'Type', 'bhp', 'Val', 100 * barsa, 'Radius', 0.1, 'Comp_i', 1);
state = incompTPFA(initState(G, W, 0), G, hT, fluid, 'wells', W);
p = state.pressure; flux = state.flux;
wi1 = W(1).WI; wi2 = W(2).WI; dz1 = W(1).dZ; dz2 = W(2).dZ;
wflux1 = state.wellSol(1).flux; wflux2 = state.wellSol(2).flux;
bhp1 = state.wellSol(1).pressure; bhp2 = state.wellSol(2).pressure;
save(fullfile(thisDir, 'ref_p2_wells.mat'), '-v7', 'p', 'flux', 'hT', ...
     'wi1', 'wi2', 'dz1', 'dz2', 'wflux1', 'wflux2', 'bhp1', 'bhp2');

%% Case 4: hydrostatic gravity column (1x1x30)
G = computeGeometry(cartGrid([1, 1, 30], [1, 1, 300]));
rock  = makeRock(G, 100 * milli * darcy, 0.2);
hT    = computeTrans(G, rock);
bd    = any(G.faces.neighbors == 0, 2);
top   = find(bd & G.faces.centroids(:, 3) < 1e-10);
bc    = addBC([], top, 'pressure', 100 * barsa);
state = incompTPFA(initResSol(G, 0), G, hT, fluid, 'bc', bc);
p = state.pressure; flux = state.flux;
save(fullfile(thisDir, 'ref_p2_gravity.mat'), '-v7', 'p', 'flux', 'hT');

gravity reset; gravity off
fprintf('Wrote 4 Phase-2 reference files to %s\n', thisDir);
