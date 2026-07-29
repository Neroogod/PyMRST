%% Dump Phase 411 reference: 3-D first-order VEM inner product, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p411_vem_ip_3d.mat.
%
% Pins PyMRST's 3-D k=1 `compute_virtual_ip` against MRST's `computeVirtualIP`
% (k=1) on a twisted 3-D grid with a full permeability tensor. Per-cell local
% stiffness blocks (node dofs) are compared by their sorted eigenvalues - an
% ordering-independent fingerprint that pins the per-face 2-D projection
% (PiNFstar), the vertex-average constant row and the ip_simple stability.

mrstModule add vem vemmech

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
perm = repmat([2.0, 0.5, 0.3, 1.5, 0.2, 1.0], nc, 1) * darcy;   % full 3-D tensor
rock = makeRock(G, perm, 0.2);

S = computeVirtualIP(G, rock, 1);

NP = diff(G.cells.nodePos);
pos = [0; cumsum(NP)];
eigblk = zeros(nc, max(NP));
for cc = 1:nc
    b = full(S.A(pos(cc)+1:pos(cc+1), pos(cc)+1:pos(cc+1)));
    eigblk(cc, 1:NP(cc)) = sort(eig((b+b')/2))';
end

save(fullfile(thisDir, 'ref_p411_vem_ip_3d.mat'), '-v7', 'eigblk', 'NP');

fprintf('Wrote ref_p411_vem_ip_3d.mat; %d cells; eig range [%.4e, %.4e]\n', ...
        nc, min(eigblk(:)), max(eigblk(:)));
