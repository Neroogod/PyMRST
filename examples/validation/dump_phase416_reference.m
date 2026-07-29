%% Dump Phase 416 reference: 3-D k=2 incompVEM standard BC + Neumann, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p416_incompvem_3d_k2_bc.mat.
%
% Pins the PyMRST 3-D k=2 standard boundary-condition path: per-face-constant
% Dirichlet on left/right, a prescribed Neumann flux on the bottom, natural
% no-flow elsewhere, on a planar-faced grid (where k=2 3-D is exact). Exercises
% the Neumann RHS at the face-moment dof and the Dirichlet imposition over
% node/edge/face dofs.

mrstModule add vem vemmech incomp

thisDir = fileparts(mfilename('fullpath'));

nx = 3; L = nx;
G = cartGrid([nx, nx, nx], [L, L, L]);
G = computeVEMGeometry(G);

nc = G.cells.num;
rock = makeRock(G, repmat([2.0, 0.5, 0.3, 1.5, 0.2, 1.0], nc, 1) * darcy, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

S = computeVirtualIP(G, rock, 2);

fc = G.faces.centroids; tol = 1e-6;
left   = find(abs(fc(:,1) - min(fc(:,1))) < tol);
right  = find(abs(fc(:,1) - max(fc(:,1))) < tol);
bottom = find(abs(fc(:,3) - min(fc(:,3))) < tol);
bc = addBC([], left,   'pressure', 1.0);
bc = addBC(bc, right,  'pressure', 0.0);
bc = addBC(bc, bottom, 'flux',     0.3 * darcy);

state = incompVEM(initState(G, [], 0), G, S, fluid, 'bc', bc);

nodePressure = state.nodePressure;
edgePressure = state.edgePressure;
facePressure = state.facePressure;
cellPressure = state.pressure;
flux         = full(state.flux);

save(fullfile(thisDir, 'ref_p416_incompvem_3d_k2_bc.mat'), '-v7', ...
     'nodePressure', 'edgePressure', 'facePressure', 'cellPressure', 'flux');

fprintf('Wrote ref_p416_incompvem_3d_k2_bc.mat\n');
