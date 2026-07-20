%% Dump Phase 6 reference: processGRDECL on three corner-point grids.
%
% Cases: the SPE1 GRDECL (regular), a faulted simpleGrdecl, and the
% Johansen NPD5 sector (real field: faults, ACTNUM, pinch-outs). For each
% we save the raw grdecl arrays plus order-independent invariants of the
% processed grid: volumes/centroids by indexMap and per-cell-pair summed
% connection areas and TPFA transmissibilities.

thisDir = fileparts(mfilename('fullpath'));

% Case 1: SPE1 GRDECL (regular corner-point grid)
g1 = readGRDECL(fullfile(ROOTDIR, 'examples', 'data', 'SPE1', ...
                         'SPE1.GRDECL'));
dump_case(g1, fullfile(thisDir, 'ref_p6_spe1.mat'));

% Case 2: synthetic faulted grid
g2 = simpleGrdecl([20, 10, 5], 0.15, 'physDims', [500, 300, 60]);
dump_case(g2, fullfile(thisDir, 'ref_p6_fault.mat'));

% Case 3: Johansen NPD5 sector model
g3 = readGRDECL(fullfile(ROOTDIR, 'examples', 'data', 'Johansen', ...
                         'NPD5.grdecl'));
g3.ACTNUM = load(fullfile(ROOTDIR, 'examples', 'data', 'Johansen', ...
                          'NPD5_ActiveCells.txt'));
dump_case(g3, fullfile(thisDir, 'ref_p6_johansen.mat'));

fprintf('Wrote 3 Phase-6 reference files to %s\n', thisDir);

function dump_case(grdecl, fname)
   G = processGRDECL(grdecl, 'SplitDisconnected', false);
   G = computeGeometry(G);
   rock = struct('perm', ones(G.cells.num, 1), 'poro', ...
                 0.3 * ones(G.cells.num, 1));
   hT = computeTrans(G, rock);
   cf = G.cells.faces(:, 1);
   T  = 1 ./ accumarray(cf, 1 ./ hT, [G.faces.num, 1]);

   N        = G.faces.neighbors;
   internal = all(N ~= 0, 2);
   imap     = G.cells.indexMap;
   pair     = sort([imap(N(internal, 1)), imap(N(internal, 2))], 2);
   area     = G.faces.areas(internal);
   Tint     = T(internal);

   [upair, ~, ic] = unique(pair, 'rows');
   conn_area  = accumarray(ic, area);
   conn_trans = accumarray(ic, Tint);
   conn_count = accumarray(ic, 1);

   cells_num = G.cells.num; faces_num = G.faces.num;
   nodes_num = G.nodes.num;
   indexMap  = G.cells.indexMap;
   volumes   = G.cells.volumes;
   centroids = G.cells.centroids;
   total_bnd_area = sum(G.faces.areas(~internal));
   cartDims  = G.cartDims;
   COORD = grdecl.COORD; ZCORN = grdecl.ZCORN;
   if isfield(grdecl, 'ACTNUM'), ACTNUM = grdecl.ACTNUM;
   else, ACTNUM = ones(prod(grdecl.cartDims), 1); end

   save(fname, '-v7', 'cells_num', 'faces_num', 'nodes_num', ...
        'indexMap', 'volumes', 'centroids', 'upair', 'conn_area', ...
        'conn_trans', 'conn_count', 'total_bnd_area', 'cartDims', ...
        'COORD', 'ZCORN', 'ACTNUM');
end
