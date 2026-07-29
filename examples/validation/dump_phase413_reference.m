%% Dump Phase 413 reference: 3-D second-order VEM inner product, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p413_vem_ip_3d_k2.mat.
%
% Pins the PyMRST 3-D k=2 `compute_virtual_ip` against MRST's `computeVirtualIP`
% (k=2) on a twisted 3-D grid with a full permeability tensor. Per-cell local
% stiffness blocks (dofs = nodes + edges + faces + one cell moment) are compared
% by sorted eigenvalues - an ordering-independent fingerprint that pins the face
% k=2 projections, the quadratic consistency term and the ip_simple stability.

mrstModule add vem vemmech

thisDir = fileparts(mfilename('fullpath'));
nx = 3; L = nx; amp = 0.2;
perm = repmat([2.0, 0.5, 0.3, 1.5, 0.2, 1.0], nx^3, 1) * darcy;

% Planar-faced grid: the k=2 3-D method is exact here (machine-precision pin).
Gp = computeVEMGeometry(cartGrid([nx, nx, nx], [L, L, L]));
Sp = computeVirtualIP(Gp, makeRock(Gp, perm, 0.2), 2);
NP = diff(Gp.cells.nodePos) + diff(Gp.cells.edgePos) + diff(Gp.cells.facePos) + 1;
pos = [0; cumsum(NP)];
eigblk_planar = zeros(Gp.cells.num, max(NP));
for cc = 1:Gp.cells.num
    b = full(Sp.A(pos(cc)+1:pos(cc+1), pos(cc)+1:pos(cc+1)));
    eigblk_planar(cc, 1:NP(cc)) = sort(eig((b+b')/2))';
end

% Twisted (non-planar-faced) grid: characterisation only. MRST integrates the
% warped faces with an edge Gauss-Lobatto rule using the projected 3-D edge
% normals; PyMRST integrates the flattened polygon exactly, so the two
% convergent schemes differ by O(twist) here.
Gt = cartGrid([nx, nx, nx], [L, L, L]);
c = Gt.nodes.coords; x = c(:,1); y = c(:,2); z = c(:,3);
c(:,1) = c(:,1) + amp*sin(pi*x/L).*sin(pi*y/L);
c(:,2) = c(:,2) + 0.75*amp*sin(pi*x/L).*sin(pi*z/L);
c(:,3) = c(:,3) + 0.5*amp*sin(pi*y/L).*sin(pi*z/L);
Gt.nodes.coords = c;
Gt = computeVEMGeometry(Gt);
St = computeVirtualIP(Gt, makeRock(Gt, perm, 0.2), 2);
pos = [0; cumsum(NP)];
eigblk_twisted = zeros(Gt.cells.num, max(NP));
for cc = 1:Gt.cells.num
    b = full(St.A(pos(cc)+1:pos(cc+1), pos(cc)+1:pos(cc+1)));
    eigblk_twisted(cc, 1:NP(cc)) = sort(eig((b+b')/2))';
end

save(fullfile(thisDir, 'ref_p413_vem_ip_3d_k2.mat'), '-v7', ...
     'eigblk_planar', 'eigblk_twisted');

fprintf('Wrote ref_p413_vem_ip_3d_k2.mat; planar+twisted %d cells\n', Gp.cells.num);
