%% Dump Phase 407 reference: incompVEM standard-BC + Neumann (k=1, 2D), MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p407_incompvem_bc.mat.
%
% Complements Phase 406 (function-handle Dirichlet + source) by pinning the
% standard MRST boundary-condition path: per-face-constant Dirichlet on the left
% and right, a prescribed Neumann flux on the bottom, natural no-flow on top.
% This exercises the Neumann RHS assembly and the boundary-flux handling that
% the func-BC path bypasses.

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

S = computeVirtualIP(G, rock, 1);

fc = G.faces.centroids;
tol = 1e-6;
left   = find(abs(fc(:,1) - min(fc(:,1))) < tol);
right  = find(abs(fc(:,1) - max(fc(:,1))) < tol);
bottom = find(abs(fc(:,2) - min(fc(:,2))) < tol);

bc = addBC([], left,   'pressure', 1.0);
bc = addBC(bc, right,  'pressure', 0.0);
bc = addBC(bc, bottom, 'flux',     0.3 * darcy);

state0 = initState(G, [], 0);
state  = incompVEM(state0, G, S, fluid, 'bc', bc);

nodePressure = state.nodePressure;
flux         = full(state.flux);
leftF   = left;
rightF  = right;
bottomF = bottom;

save(fullfile(thisDir, 'ref_p407_incompvem_bc.mat'), '-v7', ...
     'nodePressure', 'flux', 'leftF', 'rightF', 'bottomF');

fprintf('Wrote ref_p407_incompvem_bc.mat\n');
