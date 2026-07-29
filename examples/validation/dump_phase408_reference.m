%% Dump Phase 408 reference: second-order (k=2) VEM inner product, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p408_vem_ip_k2.mat.
%
% Pins the PyMRST k=2 `compute_virtual_ip` against MRST's `computeVirtualIP`
% (k=2) on a smoothly-twisted grid with a full permeability tensor. Per-cell
% local-stiffness blocks (dofs = nodes + faces + one cell moment) are compared
% by their sorted eigenvalues, which pin the quadratic consistency term and the
% ip_simple stability independently of the dof ordering.

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
perm = repmat([2.0, 0.6, 1.0], nc, 1) * darcy;
rock = makeRock(G, perm, 0.2);

S = computeVirtualIP(G, rock, 2);

NP = diff(G.cells.nodePos) + diff(G.cells.facePos) + 1;   % k=2 dofs per cell
pos = [0; cumsum(NP)];
eigblk = zeros(nc, max(NP));
for cc = 1:nc
    b = full(S.A(pos(cc)+1:pos(cc+1), pos(cc)+1:pos(cc+1)));
    eigblk(cc, 1:NP(cc)) = sort(eig((b+b')/2))';
end

save(fullfile(thisDir, 'ref_p408_vem_ip_k2.mat'), '-v7', 'eigblk', 'NP');

fprintf('Wrote ref_p408_vem_ip_k2.mat; %d cells; eig range [%.4e, %.4e]\n', ...
        nc, min(eigblk(:)), max(eigblk(:)));
