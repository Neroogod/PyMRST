%% Export advanced addWell cases for PyMRST params validation.

thisDir = fileparts(mfilename('fullpath'));
gravity reset; gravity off

G = computeGeometry(cartGrid([2, 2, 2], [20, 20, 20]));
rock = makeRock(G, repmat([100, 50, 10]*milli*darcy, G.cells.num, 1), 0.2);

%% Logical cells, partial WI, structure options, and representative radius.
mask = false(G.cells.num, 1);
mask([1, 3, 5]) = true;
W = addWell([], G, rock, mask, ...
    'Type', 'rate', 'Val', -2, 'Sign', 1, ...
    'Radius', [0.10; 0.12; 0.14], 'Skin', [0; 1; 2], ...
    'WI', [12.5; -1; 7.5], 'status', false, ...
    'vfp_index', 3, 'cp', [4; 5; 6], ...
    'components', [0.2, 0.8], 'heel', [1, 2, 3], ...
    'lims', struct('bhp', 1e7));

cells = W.cells;
WI = W.WI;
r = W.r;
rR = W.rR;
dZ = W.dZ;
signValue = W.sign;
statusValue = W.status;
cstatus = W.cstatus;
vfpIndex = W.vfp_index;
cp = W.cp;
components = W.components;
heel = W.heel;
limitBhp = W.lims.bhp;
save(fullfile(thisDir, 'ref_params_add_well_options.mat'), '-v7', ...
    'cells', 'WI', 'r', 'rR', 'dZ', 'signValue', 'statusValue', ...
    'cstatus', 'vfpIndex', 'cp', 'components', 'heel', 'limitBhp');

%% Slanted trajectory with explicit cell dimensions.
trajectoryCells = [1; 5];
[dx, dy, dz] = cellDims(G, trajectoryCells);
dims = [dx, dy, dz];
segments = [3, 4, 5; 5, 2, 1];
W = addWell([], G, rock, trajectoryCells, ...
    'cellDims', dims, 'lineSegments', segments);
cells = W.cells;
WI = W.WI;
rR = W.rR;
Wsimple = addWell([], G, rock, trajectoryCells, ...
    'InnerProduct', 'ip_simple');
Wquasitpf = addWell([], G, rock, trajectoryCells, ...
    'InnerProduct', 'ip_quasitpf');
WI_simple = Wsimple.WI;
WI_quasitpf = Wquasitpf.WI;
save(fullfile(thisDir, 'ref_params_add_well_segments.mat'), '-v7', ...
    'cells', 'dims', 'segments', 'WI', 'rR', 'WI_simple', 'WI_quasitpf');

%% Explicitly disabled representative-radius calculation.
W = addWell([], G, rock, 1, 'calcReprRad', false);
WI = W.WI;
rR = W.rR;
save(fullfile(thisDir, 'ref_params_add_well_no_repr.mat'), '-v7', 'WI', 'rR');

fprintf('Wrote advanced addWell references to %s\n', thisDir);
