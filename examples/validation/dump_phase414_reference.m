%% Dump Phase 414 reference: 3-D second-order incompVEM (planar), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p414_incompvem_3d_k2.mat.
%
% Pins the full PyMRST 3-D k=2 `incomp_vem` (assembly over node/edge/face/cell
% dofs, source at the cell moment dof, exact-nodal Dirichlet at nodes, edges and
% face moments, and flux reconstruction) against MRST's `incompVEM` on a
% PLANAR-faced grid (where the k=2 3-D method is exact). Boundary data is a
% quadratic function handle and a balanced source pair drives the cell-dof RHS.

mrstModule add vem vemmech incomp

thisDir = fileparts(mfilename('fullpath'));

nx = 3; L = nx;
G = cartGrid([nx, nx, nx], [L, L, L]);
G = computeVEMGeometry(G);

nc = G.cells.num;
perm = repmat([2.0, 0.5, 0.3, 1.5, 0.2, 1.0], nc, 1) * darcy;
rock = makeRock(G, perm, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

S = computeVirtualIP(G, rock, 2);

gfun = @(X) X(:,1).^2 - 0.5*X(:,1).*X(:,2) + 2.0*X(:,3).^2 ...
            + 3.0*X(:,1) - X(:,2) + 0.5*X(:,3) + 4.0;
bf = boundaryFaces(G);
bc = addBCVEM([], bf, 'pressure', gfun);
src = addSource([], [5, 22], [1.0, -1.0]);

state = incompVEM(initState(G, [], 0), G, S, fluid, 'bc', bc, 'src', src);

nodePressure = state.nodePressure;
edgePressure = state.edgePressure;
facePressure = state.facePressure;
cellPressure = state.pressure;
flux         = full(state.flux);

save(fullfile(thisDir, 'ref_p414_incompvem_3d_k2.mat'), '-v7', ...
     'nodePressure', 'edgePressure', 'facePressure', 'cellPressure', 'flux');

fprintf('Wrote ref_p414_incompvem_3d_k2.mat\n');
