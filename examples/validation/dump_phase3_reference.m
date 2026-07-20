%% Dump Phase 3 transport reference solutions from MATLAB MRST.
%
% Run in MATLAB after MRST startup; validate_phase3.py compares PyMRST.
% Explicit transport uses a FIXED dt so substeps match exactly.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add incomp
gravity reset; gravity off

%% Case 1+2: 1D Buckley-Leverett, explicit (fixed dt) and implicit
G     = computeGeometry(cartGrid([100, 1, 1], [100, 1, 1]));
rock  = makeRock(G, 100 * milli * darcy, 0.2);
fluid = initSimpleFluid('mu', [1, 10] * centi * poise, ...
                        'rho', [1000, 700], 'n', [2, 2]);
hT    = computeTrans(G, rock);
rate  = sum(poreVolume(G, rock)) / (100 * day);   % 1 PV per 100 days
src   = addSource([] , 1  ,  rate, 'sat', [1, 0]);
src   = addSource(src, 100, -rate, 'sat', [0, 1]);
st0   = incompTPFA(initResSol(G, 0, [0, 1]), G, hT, fluid, 'src', src);

tf = 30 * day;
stE = explicitTransport(st0, G, tf, rock, fluid, 'src', src, ...
                        'computedt', false, 'dt', 0.25 * day);
sE  = stE.s(:, 1);
stI = implicitTransport(st0, G, tf, rock, fluid, 'src', src);
sI  = stI.s(:, 1);
p0  = st0.pressure; flux0 = st0.flux;
save(fullfile(thisDir, 'ref_p3_bl.mat'), '-v7', 'sE', 'sI', 'p0', 'flux0');

%% Case 3: quarter five-spot, two sequential pressure/transport cycles
G     = computeGeometry(cartGrid([32, 32, 1], [500, 500, 10]));
kvals = (10 + mod(7 * (0 : G.cells.num - 1)', 100)) * milli * darcy;
rock  = makeRock(G, kvals, 0.2);
hT    = computeTrans(G, rock);
rate  = 5e-3;
src   = addSource([] , 1          ,  rate, 'sat', [1, 0]);
src   = addSource(src, G.cells.num, -rate, 'sat', [0, 1]);
state = initResSol(G, 0, [0, 1]);
dT    = 100 * day;
for i = 1 : 2
    state = incompTPFA(state, G, hT, fluid, 'src', src);
    state = implicitTransport(state, G, dT, rock, fluid, 'src', src);
end
sQ = state.s(:, 1); pQ = state.pressure;
save(fullfile(thisDir, 'ref_p3_qfs.mat'), '-v7', 'sQ', 'pQ');

%% Case 4: gravity segregation column (no Darcy flow), implicit
gravity reset; gravity on
G     = computeGeometry(cartGrid([1, 1, 20], [1, 1, 20]));
rock  = makeRock(G, 1000 * milli * darcy, 0.3);
fluidG = initSimpleFluid('mu', [1, 1] * centi * poise, ...
                         'rho', [1000, 700], 'n', [2, 2]);
state = initResSol(G, 0, [0.5, 0.5]);
state.flux = zeros(G.faces.num, 1);
state = implicitTransport(state, G, 50 * day, rock, fluidG);
sG = state.s(:, 1);
save(fullfile(thisDir, 'ref_p3_gravity.mat'), '-v7', 'sG');

gravity reset; gravity off
fprintf('Wrote 3 Phase-3 reference files to %s\n', thisDir);
