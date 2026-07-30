%% Reference data for computeGeometry hingenodes on a non-planar face

thisDir = fileparts(mfilename('fullpath'));
G = cartGrid([1, 1, 1], [1, 1, 1]);
face = G.cells.faces(5, 1);
ix = G.faces.nodePos(face):G.faces.nodePos(face + 1) - 1;
face_nodes = G.faces.nodes(ix);
G.nodes.coords(face_nodes(end), 3) = ...
    G.nodes.coords(face_nodes(end), 3) + 0.4;
hinge = mean(G.nodes.coords(face_nodes, :), 1) + [0.1, -0.05, 0.2];

D = computeGeometry(G);
H = computeGeometry(G, 'hingenodes', ...
                    struct('faces', face, 'nodes', hinge));

face0 = face - 1;
face_nodes0 = face_nodes - 1;
node_coords = G.nodes.coords;
default_face_areas = D.faces.areas;
default_face_normals = D.faces.normals;
default_face_centroids = D.faces.centroids;
default_cell_volumes = D.cells.volumes;
default_cell_centroids = D.cells.centroids;
hinged_face_areas = H.faces.areas;
hinged_face_normals = H.faces.normals;
hinged_face_centroids = H.faces.centroids;
hinged_cell_volumes = H.cells.volumes;
hinged_cell_centroids = H.cells.centroids;

save(fullfile(thisDir, 'ref_grid_compute_geometry_hinge.mat'), '-v7', ...
     'face0', 'face_nodes0', 'node_coords', 'hinge', ...
     'default_face_areas', 'default_face_normals', ...
     'default_face_centroids', 'default_cell_volumes', ...
     'default_cell_centroids', 'hinged_face_areas', ...
     'hinged_face_normals', 'hinged_face_centroids', ...
     'hinged_cell_volumes', 'hinged_cell_centroids');
fprintf('Wrote computeGeometry hingenodes reference to %s\n', thisDir);
