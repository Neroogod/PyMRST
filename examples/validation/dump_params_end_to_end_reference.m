%% End-to-end params workflow on an inactive corner-point grid.

mrstModule add deckformat incomp
thisDir = fileparts(mfilename('fullpath'));
gravity reset; gravity on

grdecl = simpleGrdecl([3, 2, 2], @(x) 0.12*sin(2*pi*x));
nc = prod(grdecl.cartDims);
grdecl.ACTNUM = ones(nc, 1); grdecl.ACTNUM(5) = 0;
base = (1:nc)';
grdecl.PERMX = (100 + 3*base)*milli*darcy;
grdecl.PERMY = (70 + 2*base)*milli*darcy;
grdecl.PERMZ = (10 + base)*milli*darcy;
grdecl.PORO = 0.15 + 0.005*base;
grdecl.NTG = 0.9 - 0.01*base;

G = computeGeometry(processGRDECL(grdecl));
rock = grdecl2Rock(grdecl, G);

% Mixed-direction well in selected active Cartesian cells.
targetGlobal = [1; 3; 7; 10];
[present, directCells] = ismember(targetGlobal, G.cells.indexMap);
assert(all(present));
W = addWell([], G, rock, directCells, ...
    'Type', 'rate', 'Val', 1e-5, 'Name', 'CP-I', ...
    'Dir', ['x';'y';'z';'x'], 'Radius', [0.009; 0.010; 0.011; 0.012], ...
    'Skin', [0; 0.5; 1; 1.5], 'WI', [-1; 2e-12; -1; -1], ...
    'Comp_i', 1, 'status', true);

% Logical column (I=2,J=2) has inactive top cell 5 and active lower cell 11.
W = verticalWell(W, G, rock, 2, 2, [], ...
    'Type', 'bhp', 'Val', 90*barsa, 'Name', 'CP-P', 'Comp_i', 1, ...
    'Radius', 0.01);

bc = pside([], G, 'xmin', 120*barsa, 'sat', 1);
bc = fluxside(bc, G, 'xmax', -2e-6, 'sat', 1);
sourceCell = find(G.cells.indexMap == 6);
src = addSource([], sourceCell, 1e-6, 'sat', 1);

hT = computeTrans(G, rock);
pv = poreVolume(G, rock);
fluid = initSingleFluid('mu', 1*centi*poise, 'rho', 1000);
state = incompTPFA(initState(G, W, 100*barsa), G, hT, fluid, ...
    'bc', bc, 'src', src, 'wells', W);

in_cartDims = grdecl.cartDims(:); in_COORD = grdecl.COORD(:);
in_ZCORN = grdecl.ZCORN(:); in_ACTNUM = grdecl.ACTNUM(:);
in_PERMX = grdecl.PERMX(:); in_PERMY = grdecl.PERMY(:);
in_PERMZ = grdecl.PERMZ(:); in_PORO = grdecl.PORO(:); in_NTG = grdecl.NTG(:);
indexMap = G.cells.indexMap(:);
cellCentroids = G.cells.centroids; cellVolumes = G.cells.volumes;
rockPerm = rock.perm; rockPoro = rock.poro; rockNtg = rock.ntg;
w1cells = W(1).cells; w1WI = W(1).WI; w1dZ = W(1).dZ; w1rR = W(1).rR;
w2cells = W(2).cells; w2WI = W(2).WI; w2dZ = W(2).dZ; w2rR = W(2).rR;
bcFace = bc.face; bcValue = bc.value; bcSat = bc.sat;
bcCentroids = G.faces.centroids(bc.face, :);
srcCell = src.cell; srcRate = src.rate; srcSat = src.sat;
pressure = state.pressure; flux = state.flux;
cellNo = rldecode((1:G.cells.num)', diff(G.cells.facePos));
hTSum = accumarray(cellNo, hT);
halfFaces = G.cells.faces(:,1);
sgn = 2*double(G.faces.neighbors(halfFaces,1) == cellNo)-1;
cellDivergence = accumarray(cellNo, sgn.*state.flux(halfFaces));
wflux1 = state.wellSol(1).flux; wflux2 = state.wellSol(2).flux;
bhp1 = state.wellSol(1).pressure; bhp2 = state.wellSol(2).pressure;

save(fullfile(thisDir, 'ref_params_end_to_end.mat'), '-v7', ...
    'in_cartDims','in_COORD','in_ZCORN','in_ACTNUM', ...
    'in_PERMX','in_PERMY','in_PERMZ','in_PORO','in_NTG', ...
    'indexMap','cellCentroids','cellVolumes','rockPerm','rockPoro','rockNtg', ...
    'w1cells','w1WI','w1dZ','w1rR','w2cells','w2WI','w2dZ','w2rR', ...
    'bcFace','bcCentroids','bcValue','bcSat','srcCell','srcRate','srcSat', ...
    'hT','hTSum','pv','pressure','flux','cellDivergence', ...
    'wflux1','wflux2','bhp1','bhp2');
gravity reset; gravity off
fprintf('Wrote params end-to-end reference to %s\n', thisDir);
