%% Dump Phase 405 reference: mixed VEM inner product, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p405_vem_ip.mat.
%
% Pins the PyMRST `compute_virtual_ip` against MRST's `computeVirtualIP` (k=1) on
% a smoothly-twisted grid with a full permeability tensor. The per-cell local
% stiffness blocks are compared by their sorted eigenvalues - an
% ordering-independent fingerprint of the VEM inner product that pins the
% edge-weighted constant projection, the tensor gradient rows and the ip_simple
% stability exactly.

mrstModule add vem vemmech

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

S = computeVirtualIP(G, rock, 1);

NP = diff(G.cells.nodePos);                        % nodes per cell (all 4 here)
pos = [0; cumsum(NP)];
eigblk = zeros(nc, max(NP));
for cc = 1:nc
    b = full(S.A(pos(cc)+1:pos(cc+1), pos(cc)+1:pos(cc+1)));
    eigblk(cc, 1:NP(cc)) = sort(eig((b+b')/2))';
end

save(fullfile(thisDir, 'ref_p405_vem_ip.mat'), '-v7', 'eigblk', 'NP');

fprintf('Wrote ref_p405_vem_ip.mat to %s\n', thisDir);
fprintf('  %d cells; eig range [%.4e, %.4e]\n', nc, min(eigblk(:)), max(eigblk(:)));
