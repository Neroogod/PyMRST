%% Dump Phase 410 reference: VEM grid geometry (edges/diameters), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p410_vem_geometry.mat.
%
% Pins PyMRST `compute_vem_geometry` against MRST's `computeVEMGeometry` on a
% twisted 3-D grid: the deduplicated edge list, per-face edge maps/signs and
% in-plane edge normals, cell edges, and cell/face diameters.

mrstModule add vem vemmech

thisDir = fileparts(mfilename('fullpath'));

nx = 3; L = nx;
G = cartGrid([nx, nx, nx], [L, L, L]);
c = G.nodes.coords;
x = c(:,1); y = c(:,2); z = c(:,3);
c(:,1) = c(:,1) + 0.2*sin(pi*x/L).*sin(pi*y/L);
c(:,2) = c(:,2) + 0.15*sin(pi*x/L).*sin(pi*z/L);
c(:,3) = c(:,3) + 0.1*sin(pi*y/L).*sin(pi*z/L);
G.nodes.coords = c;
G = computeVEMGeometry(G);

edgesNum       = G.edges.num;
edgesNodes     = G.edges.nodes;        % 1-indexed, flattened pairs
edgesLengths   = G.edges.lengths;
edgesCentroids = G.edges.centroids;
facesEdges     = G.faces.edges;        % 1-indexed
facesEdgePos   = G.faces.edgePos;
facesEdgeSign  = G.faces.edgeSign;
facesEdgeNorm  = G.faces.edgeNormals;
cellsEdges     = G.cells.edges;        % 1-indexed
cellsEdgePos   = G.cells.edgePos;
cellDiameters  = G.cells.diameters;
faceDiameters  = G.faces.diameters;

save(fullfile(thisDir, 'ref_p410_vem_geometry.mat'), '-v7', ...
     'edgesNum', 'edgesNodes', 'edgesLengths', 'edgesCentroids', ...
     'facesEdges', 'facesEdgePos', 'facesEdgeSign', 'facesEdgeNorm', ...
     'cellsEdges', 'cellsEdgePos', 'cellDiameters', 'faceDiameters');

fprintf('Wrote ref_p410_vem_geometry.mat; %d edges, %d faces, %d cells\n', ...
        edgesNum, G.faces.num, G.cells.num);
