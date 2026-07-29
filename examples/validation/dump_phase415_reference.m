%% Dump Phase 415 reference: incompVEM with TPFA flux reconstruction, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p415_incompvem_tpfa.mat.
%
% Pins PyMRST's `trans='tpfa'` flux-reconstruction option against MRST's
% `incompVEM` with `computeVirtualIP(G, rock, k, 'trans', 'tpfa')` on a twisted
% 2-D grid: instead of the direct VEM gradient, fluxes come from a TPFA scheme
% on the reconstructed cell/face pressures. Dumps node pressures and fluxes for
% k = 1 and k = 2 (Dirichlet left/right, natural no-flow top/bottom).

mrstModule add vem vemmech incomp

thisDir = fileparts(mfilename('fullpath'));

nx = 6; L = nx;
G = cartGrid([nx, nx], [L, L]);
c = G.nodes.coords; x = c(:,1); y = c(:,2);
c(:,1) = c(:,1) + 0.3*sin(pi*x/L).*sin(pi*y/L);
c(:,2) = c(:,2) + 0.6*0.3*sin(2*pi*x/L).*sin(pi*y/L);
G.nodes.coords = c;
G = computeVEMGeometry(G);

nc = G.cells.num;
rock = makeRock(G, repmat([2.0, 0.6, 1.0], nc, 1) * darcy, 0.2);
fluid = initSingleFluid('mu', 1, 'rho', 1);

fc = G.faces.centroids; tol = 1e-6;
left  = find(abs(fc(:,1) - min(fc(:,1))) < tol);
right = find(abs(fc(:,1) - max(fc(:,1))) < tol);
bc = addBC([], left, 'pressure', 1.0);
bc = addBC(bc, right, 'pressure', 0.0);

for k = 1:2
    S = computeVirtualIP(G, rock, k, 'trans', 'tpfa');
    state = incompVEM(initState(G, [], 0), G, S, fluid, 'bc', bc);
    eval(sprintf('nodePressure%d = state.nodePressure;', k));
    eval(sprintf('flux%d = full(state.flux);', k));
end

save(fullfile(thisDir, 'ref_p415_incompvem_tpfa.mat'), '-v7', ...
     'nodePressure1', 'flux1', 'nodePressure2', 'flux2');

fprintf('Wrote ref_p415_incompvem_tpfa.mat\n');
