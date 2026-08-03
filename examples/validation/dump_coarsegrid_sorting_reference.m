% Reference for partitionTensor and sortSubEdges.
%
% Run from the MRST root:
%   matlab -batch "startup; run('.../dump_coarsegrid_sorting_reference.m')"

mrstModule add coarsegrid;

out = struct();

% ------------------------------------------------ case A: partitionTensor 3D
G = computeGeometry(cartGrid([6, 4, 2]));
out.A = struct('p_uneven', partitionTensor(G, [1 4 1], [4], [2]), ...
               'p_even',   partitionTensor(G, [3 3], [2 2], [2]), ...
               'p_unit',   partitionTensor(G, ones(1,6), ones(1,4), ones(1,2)));

% --------------------------------- case B: partitionTensor with a hole in it
G = computeGeometry(cartGrid([6, 4, 1]));
G = removeCells(G, [8 9 14 15]);
out.B = struct('p', partitionTensor(G, [3 3], [2 2], [1]), ...
               'indexMap', G.cells.indexMap);

% -------------------------------------------------- case C: partitionTensor 2D
G = computeGeometry(cartGrid([6, 4]));
out.C = struct('p', partitionTensor(G, [2 4], [1 3]));

% ------------------------------------------------------ case D: sortSubEdges
G  = computeGeometry(cartGrid([6, 6]));
p  = partitionCartGrid([6 6 1], [2 2 1]);
CG = generateCoarseGrid(G, p);
out.D = struct('p', p, ...
               'fconn_before', CG.faces.fconn, ...
               'connPos',      CG.faces.connPos, ...
               'facePos',      CG.cells.facePos, ...
               'faces_before', CG.cells.faces(:,1), ...
               'neighbors',    CG.faces.neighbors);
CGs = sortSubEdges(CG);
out.D.fconn_after = CGs.faces.fconn;
out.D.faces_after = CGs.cells.faces(:,1);

dest = fullfile(fileparts(mfilename('fullpath')), 'ref_coarsegrid_sorting.mat');
save(dest, '-struct', 'out', '-v7');
fprintf('wrote %s\n', dest);
