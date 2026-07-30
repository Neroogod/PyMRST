%% Reference data for processGRDECLOMO compatibility entry point

thisDir = fileparts(mfilename('fullpath'));
grdecl = cartesianGrdecl(0:3, 0:2, 0:2);
grdecl.ACTNUM = ones(12, 1);
grdecl.ACTNUM(6) = 0;
G = computeGeometry(processGRDECLOMO(grdecl, 'PreserveCpNodes', true), ...
                    'CpGeometry', true);

[sorted_centroids, order] = sortrows(G.cells.centroids);
sorted_volumes = G.cells.volumes(order);
cell_count = G.cells.num;
face_count = G.faces.num;
node_count = G.nodes.num;
index_map = G.cells.indexMap;
total_volume = sum(G.cells.volumes);
cpnodes = G.cells.cpnodes;
cpnode_coords = reshape(G.nodes.coords(cpnodes(:), :), ...
                        [size(cpnodes), size(G.nodes.coords, 2)]);
cp_centroids = G.cells.cpgeometry.centroids;
cp_extent = G.cells.cpgeometry.extent;
cp_facecentroids = G.cells.cpgeometry.facecentroids;

save(fullfile(thisDir, 'ref_grid_process_grdecl_omo.mat'), '-v7', ...
     'sorted_centroids', 'sorted_volumes', 'cell_count', 'face_count', ...
     'node_count', 'index_map', 'total_volume', 'cpnodes', 'cpnode_coords', ...
     'cp_centroids', 'cp_extent', 'cp_facecentroids');
fprintf('Wrote processGRDECLOMO reference to %s\n', thisDir);
