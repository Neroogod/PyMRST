% Reference for the coarse-grid face topology.
%
% Covers generateCoarseGrid's full contract, including the optional face
% partition that splits a block pair into several coarse faces, and
% processFacePartition, which makes each coarse face a connected collection of
% fine faces (MRST implements it for 2D only).
%
% Run from the MRST root:
%   matlab -batch "startup; run('.../dump_coarsegrid_faces_reference.m')"

mrstModule add coarsegrid;

out = struct();

% ---------------------------------------------------------------- case A: 3D
% Default indicator. The exterior half-face tag is part of it, so the boundary
% faces of a block do NOT collapse into one coarse face.
G = computeGeometry(cartGrid([6, 4, 2]));
p = partitionUI(G, [3, 2, 1]);
CG = generateCoarseGrid(G, p);
out.A = pack(G, p, CG);

% ---------------------------------------------------------------- case B: 2D
G = computeGeometry(cartGrid([4, 4]));
p = partitionUI(G, [2, 2]);
CG = generateCoarseGrid(G, p);
out.B = pack(G, p, CG);

% ------------------------------------------- case C: 2D with a face partition
% showCoarseGrid.m, second example: a face indicator unrelated to the cell
% partition, then processFacePartition to make each coarse face connected.
G  = computeGeometry(cartGrid([8, 8], [1 1]));
f  = @(c) sin(3*pi*(c(:,1)-c(:,2)));
pfc = 1 + (f(G.cells.centroids) > 0);
pv = partitionCartGrid(G.cartDims, [2 2]);
pf0 = cellPartitionToFacePartition(G, pfc);
pf1 = processFacePartition(G, pv, pf0);
CG = generateCoarseGrid(G, pv, pf1);
out.C = pack(G, pv, CG);
out.C.pf_cell = pfc;
out.C.pf_in   = pf0;
out.C.pf_out  = pf1;

% ------------------------- case D: 3D, face partition from a finer partition
G = computeGeometry(cartGrid([20 20 6]));
c = G.cells.centroids;
G = removeCells(G, (c(:,1)<10) & (c(:,2)<10) & (c(:,3)<3));
p = partitionUI(G, [2, 2, 2]);
q = partitionUI(G, [4, 4, 2]);
pf = cellPartitionToFacePartition(G, q);
CG = generateCoarseGrid(G, p, pf);
out.D = pack(G, p, CG);
out.D.pf_in = pf;
out.D.q = q;

% ------------------------------- case E: processFacePartition on its own, 2D
% A partition whose induced face partition already splits into disconnected
% pieces, so the function has something to do.
G = computeGeometry(cartGrid([6, 6], [1 1]));
pv = partitionCartGrid(G.cartDims, [2 2]);
cc = G.cells.centroids;
pfc = 1 + double((cc(:,1) < 0.5) == (cc(:,2) < 0.5));   % checkerboard of halves
pf0 = cellPartitionToFacePartition(G, pfc);
out.E = struct('pf_cell', pfc, 'pv', pv, 'pf_in', pf0, ...
               'pf_out', processFacePartition(G, pv, pf0));

dest = fullfile(fileparts(mfilename('fullpath')), 'ref_coarsegrid_faces.mat');
save(dest, '-struct', 'out', '-v7');
fprintf('wrote %s\n', dest);

function s = pack(G, p, CG)
   s = struct();
   s.p            = p;
   s.cells_num    = CG.cells.num;
   s.cells_facePos = CG.cells.facePos;
   s.cells_faces  = CG.cells.faces;
   s.faces_num    = CG.faces.num;
   s.faces_neighbors = CG.faces.neighbors;
   s.faces_connPos = CG.faces.connPos;
   s.faces_fconn  = CG.faces.fconn;
   s.G_faces_neighbors = G.faces.neighbors;
   s.G_cells_faces = G.cells.faces;
   s.G_cells_facePos = G.cells.facePos;
   s.G_cartDims   = G.cartDims;
   if isfield(G.cells, 'indexMap')
      s.G_indexMap = G.cells.indexMap;
   end
end
