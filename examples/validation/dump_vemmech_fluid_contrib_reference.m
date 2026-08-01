%% Reference data for addFluidContribMechVEM (poroelastic boundary traction)

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

%% 3D case: pressure on the top boundary of a unit cube.
G = createAugmentedGrid(computeGeometry(cartGrid([4, 4, 4], [1, 1, 1])));
isBoundary = any(G.faces.neighbors == 0, 2);
topFaces = find(isBoundary & abs(G.faces.centroids(:, 3) - 1) < 1e-12);
rock = struct('alpha', ones(G.cells.num, 1));
isdirdofs = false(G.griddim * G.nodes.num, 1);

bc3 = addBC([], topFaces, 'pressure', 1.0);
fbc3_top = addFluidContribMechVEM(G, bc3, rock, isdirdofs);

% every outer face at once: a closed pressurized boundary
allBoundary = find(isBoundary);
bc3all = addBC([], allBoundary, 'pressure', 1.0);
fbc3_all = addFluidContribMechVEM(G, bc3all, rock, isdirdofs);

% alpha scaling
rock_half = struct('alpha', 0.5 * ones(G.cells.num, 1));
fbc3_half = addFluidContribMechVEM(G, bc3, rock_half, isdirdofs);

% with some displacement dofs eliminated
isdir_some = false(G.griddim * G.nodes.num, 1);
isdir_some(1:9) = true;
fbc3_dir = addFluidContribMechVEM(G, bc3, rock, isdir_some);

top_faces0 = topFaces - 1;
all_boundary0 = allBoundary - 1;
node_coords3 = G.nodes.coords;

%% 2D case: pressure on the right boundary of a unit square.
G2 = createAugmentedGrid(computeGeometry(cartGrid([4, 4], [1, 1])));
isBoundary2 = any(G2.faces.neighbors == 0, 2);
rightFaces = find(isBoundary2 & abs(G2.faces.centroids(:, 1) - 1) < 1e-12);
rock2 = struct('alpha', ones(G2.cells.num, 1));
isdirdofs2 = false(G2.griddim * G2.nodes.num, 1);
bc2 = addBC([], rightFaces, 'pressure', 1.0);
fbc2_right = addFluidContribMechVEM(G2, bc2, rock2, isdirdofs2);
right_faces0 = rightFaces - 1;
node_coords2 = G2.nodes.coords;

%% Empty boundary condition
fbc3_empty = addFluidContribMechVEM(G, [], rock, isdirdofs);

save(fullfile(thisDir, 'ref_vemmech_fluid_contrib.mat'), '-v7', ...
     'node_coords3', 'top_faces0', 'all_boundary0', ...
     'fbc3_top', 'fbc3_all', 'fbc3_half', 'fbc3_dir', 'fbc3_empty', ...
     'node_coords2', 'right_faces0', 'fbc2_right');
fprintf('Wrote addFluidContribMechVEM reference to %s\n', thisDir);
