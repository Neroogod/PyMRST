%% Dump Phase 409 reference: second-order incompVEM (k=2, 2D), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p409_incompvem_k2.mat.
%
% Pins the full PyMRST k=2 `incomp_vem` (assembly over node/face/cell dofs,
% source at the cell moment dof, exact-nodal Dirichlet at nodes and face
% centroids, and flux reconstruction) against MRST's `incompVEM` on a twisted
% grid with a full permeability tensor. Boundary data is a quadratic function
% handle and a balanced source pair drives the cell-dof RHS path.

mrstModule add vem vemmech incomp

thisDir = fileparts(mfilename('fullpath'));

nx = 6; L = nx;
G = cartGrid([nx, nx], [L, L]);
c = G.nodes.coords;
x = c(:,1); y = c(:,2);
c(:,1) = c(:,1) + 0.3*sin(pi*x/L).*sin(pi*y/L);
c(:,2) = c(:,2) + 0.6*0.3*sin(2*pi*x/L).*sin(pi*y/L);
G.nodes.coords = c;
G = computeGeometry(G);
G = computeVEMGeometry(G);

nc = G.cells.num;
perm = repmat([2.0, 0.6, 1.0], nc, 1) * darcy;
rock = makeRock(G, perm, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

S = computeVirtualIP(G, rock, 2);

gfun = @(X) X(:,1).^2 - 0.5*X(:,1).*X(:,2) + 2.0*X(:,2).^2 ...
            + 3.0*X(:,1) - X(:,2) + 4.0;
bf = boundaryFaces(G);
bc = addBCVEM([], bf, 'pressure', gfun);
src = addSource([], [12, 23], [1.0, -1.0]);

state0 = initState(G, [], 0);
state  = incompVEM(state0, G, S, fluid, 'bc', bc, 'src', src);

nodePressure = state.nodePressure;
facePressure = state.facePressure;
cellPressure = state.pressure;
flux         = full(state.flux);

save(fullfile(thisDir, 'ref_p409_incompvem_k2.mat'), '-v7', ...
     'nodePressure', 'facePressure', 'cellPressure', 'flux');

fprintf('Wrote ref_p409_incompvem_k2.mat\n');
