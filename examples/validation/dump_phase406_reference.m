%% Dump Phase 406 reference: first-order incompVEM (k=1, 2D), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p406_incompvem.mat.
%
% Pins PyMRST `incomp_vem` (full solver: assembly, source, exact-nodal
% Dirichlet, direct flux reconstruction) against MRST's `incompVEM` on the same
% smoothly-twisted grid with a full permeability tensor. Boundary data is a
% bilinear function handle (addBCVEM) so both codes impose the identical exact
% nodal Dirichlet condition, and a balanced source pair exercises the RHS path.

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
perm = repmat([2.0, 0.6, 1.0], nc, 1) * darcy;     % full tensor kxx,kxy,kyy
rock = makeRock(G, perm, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

S = computeVirtualIP(G, rock, 1);

%   Exact-nodal Dirichlet (bilinear, not reproduced by k=1 => real solve).
gfun = @(X) 2.0 + 3.0*X(:,1) - 1.5*X(:,2) + 0.5*X(:,1).*X(:,2);
bf = boundaryFaces(G);
bc = addBCVEM([], bf, 'pressure', gfun);

%   Balanced source pair.
src = addSource([], [12, 23], [1.0, -1.0]);

state0 = initState(G, [], 0);
state  = incompVEM(state0, G, S, fluid, 'bc', bc, 'src', src);

nodePressure = state.nodePressure;
flux         = state.flux;

save(fullfile(thisDir, 'ref_p406_incompvem.mat'), '-v7', ...
     'nodePressure', 'flux');

fprintf('Wrote ref_p406_incompvem.mat to %s\n', thisDir);
fprintf('  %d nodes, %d faces; p range [%.4e, %.4e], flux range [%.4e, %.4e]\n', ...
        G.nodes.num, G.faces.num, min(nodePressure), max(nodePressure), ...
        min(flux), max(flux));
