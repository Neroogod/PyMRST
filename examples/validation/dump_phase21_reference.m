% Phase 21 reference: node-based trap analysis (co2lab-spillpoint/node_based)
%
% trapAnalysis(Gt, false) uses the node (edge) based method. Dumps both the
% raw node-level result (computeNodeTraps) and its projection onto cells.

mrstModule add co2lab-common co2lab-spillpoint ad-core

here = fileparts(mfilename('fullpath'));

nx = 60; ny = 44;
G = cartGrid([nx, ny, 1], [800, 600, 20]);
x = G.nodes.coords(:,1); y = G.nodes.coords(:,2);
G.nodes.coords(:,3) = G.nodes.coords(:,3) + ...
    (1000 - 0.02*x ...
     - 10*exp(-(((x - 250)/80).^2  + ((y - 300)/90).^2)) ...
     - 14*exp(-(((x - 600)/85).^2  + ((y - 300)/90).^2)));
Gt = topSurfaceGrid(computeGeometry(G));

% node-level
nres = computeNodeTraps(Gt, [], []);

% cell-level (the projection)
cres = trapAnalysis(Gt, false);

ref = struct();
% Node *indices* are internal bookkeeping and MRST's ordering is not
% PyMRST's, so the coordinates go too and the comparison is made by position.
ref.node_xy       = Gt.nodes.coords;
ref.node_z        = Gt.nodes.z(:);
ref.cell_z        = Gt.cells.z(:);
ref.n_traps       = double(max(nres.traps));
ref.node_traps    = double(nres.traps(:));
ref.node_regions  = double(nres.trap_regions(:));
ref.node_trap_z   = nres.trap_zvals(:);
ref.node_dstr     = double(nres.dstr_neigh(:));
ref.cell_traps    = double(cres.traps(:));
ref.cell_trap_z   = cres.trap_z(:);
ref.cell_regions  = double(cres.trap_regions(:));

save(fullfile(here, 'ref_p21_node_trapping.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-21 reference: %d node traps (%d trapped nodes), ' ...
         '%d cell traps (%d trapped cells)\n'], max(nres.traps), ...
        sum(nres.traps > 0), max(cres.traps), sum(cres.traps > 0));
