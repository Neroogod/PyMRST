%% Dump Phase 412 reference: 3-D first-order incompVEM, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p412_incompvem_3d.mat.
%
% Pins the full PyMRST 3-D k=1 `incomp_vem` (assembly, source, exact-nodal
% Dirichlet, flux reconstruction) against MRST's `incompVEM` on a twisted 3-D
% grid with a full permeability tensor. Both use the same exact-nodal Dirichlet
% (function handle) and a balanced source pair, so they solve the identical
% discrete system - the comparison is number-for-number, independent of the
% scheme's own (approximate) linear reproduction in 3-D.

mrstModule add vem vemmech incomp

thisDir = fileparts(mfilename('fullpath'));

nx = 3; L = nx; amp = 0.2;
G = cartGrid([nx, nx, nx], [L, L, L]);
c = G.nodes.coords;
x = c(:,1); y = c(:,2); z = c(:,3);
c(:,1) = c(:,1) + amp*sin(pi*x/L).*sin(pi*y/L);
c(:,2) = c(:,2) + 0.75*amp*sin(pi*x/L).*sin(pi*z/L);
c(:,3) = c(:,3) + 0.5*amp*sin(pi*y/L).*sin(pi*z/L);
G.nodes.coords = c;
G = computeVEMGeometry(G);

nc = G.cells.num;
perm = repmat([2.0, 0.5, 0.3, 1.5, 0.2, 1.0], nc, 1) * darcy;
rock = makeRock(G, perm, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

S = computeVirtualIP(G, rock, 1);

gfun = @(X) 1.5*X(:,1) - 2.0*X(:,2) + 0.8*X(:,3) + 3.0;
bf = boundaryFaces(G);
bc = addBCVEM([], bf, 'pressure', gfun);
src = addSource([], [5, 22], [1.0, -1.0]);

state = incompVEM(initState(G, [], 0), G, S, fluid, 'bc', bc, 'src', src);

nodePressure = state.nodePressure;
flux         = full(state.flux);

save(fullfile(thisDir, 'ref_p412_incompvem_3d.mat'), '-v7', ...
     'nodePressure', 'flux');

fprintf('Wrote ref_p412_incompvem_3d.mat\n');
