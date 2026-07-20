%% Dump Phase 78 reference: numeric VEM_assemble_AD / VEM_linElast_AD path.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p78_vemmech_ad_path.mat; then run validate_phase78.py.

mrstModule add ad-core vemmech
gravity reset on

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([1, 1, 1], [1, 1, 1]));
G = createAugmentedGrid(G);
binfo = identify_boundary_elements(G);

bcond = [0.10 0.04; ...
         0.02 0.01; ...
        -0.03 0.02; ...
         0.000 0.000; ...
         0.000 0.000];
[bcfun, num_bc] = set_bcfun(G, binfo, bcond);
u = linspace(0.2, 0.8, num_bc)';
bc_dense = dense_bcfun_reference(G, binfo, bcond, u);

E = 1.05e9 * ones(G.cells.num, 1);
nu = 0.23 * ones(G.cells.num, 1);
C = Enu2C(E, nu, G);
[S, op] = VEM_assemble(G, C);

load_vec = [0, 0, 0];
load_mat = repmat(load_vec, numel(G.cells.nodes), 1);
[uu, extra] = dense_ad_solve_reference(G, S, bc_dense);

save(fullfile(thisDir, 'ref_p78_vemmech_ad_path.mat'), '-v7', ...
     'bcond', 'num_bc', 'u', 'E', 'nu', 'S', 'uu', ...
     'load_mat', 'bc_dense', 'op', 'extra');

fprintf('Wrote ref_p78_vemmech_ad_path.mat to %s\n', thisDir);

function [uu, extra] = dense_ad_solve_reference(G, S, bc_dense)
   u_bc = zeros(G.griddim * G.nodes.num, 1);
   isdirdofs = false(G.griddim * G.nodes.num, 1);
   for n = 1:G.nodes.num
      for d = 1:G.griddim
         val = bc_dense(n, d);
         if ~isnan(val)
            dof = G.griddim * (n - 1) + d;
            u_bc(dof) = val;
            isdirdofs(dof) = true;
         end
      end
   end
   rhs0 = -S * u_bc;
   rhs = rhs0(~isdirdofs);
   A = S(~isdirdofs, ~isdirdofs);
   A = (A + A') / 2;
   x = A \ rhs;
   u = nan(G.griddim * G.nodes.num, 1);
   u(isdirdofs) = u_bc(isdirdofs);
   u(~isdirdofs) = x;
   uu = reshape(u, G.griddim, [])';
   vdiv = VEM_div(G);
   extra.A = A;
   extra.S = S;
   extra.rhs = rhs;
   extra.disc.isdirdofs = isdirdofs;
   extra.disc.rhs = rhs;
   extra.disc.V_dir = u_bc;
   extra.disc.ovol_div = vdiv;
   extra.disc.gradP = vdiv(:, ~isdirdofs)';
   extra.disc.div = vdiv(:, ~isdirdofs);
   extra.disc.divrhs = vdiv * u_bc;
end

function M = dense_bcfun_reference(G, binfo, bcond, u)
   if size(bcond, 1) == 3
      bcond = [bcond; zeros(2, 2)];
   end
   bottom_nodes = binfo.side_nodes{6};
   side_nodes = unique(vertcat(binfo.side_nodes{1:4}));
   bottom_edge_nodes = intersect(bottom_nodes, side_nodes);
   side_nodes = setdiff(side_nodes, bottom_edge_nodes);
   bottom_nodes = setdiff(bottom_nodes, bottom_edge_nodes);

   active_u = find(bcond(:, 2) > 0);
   full_u = zeros(5, 1);
   full_u(active_u) = u(1:numel(active_u));

   M = zeros(G.nodes.num, G.griddim);
   if ~isempty(bottom_nodes)
      M(bottom_nodes, 1) = nan;
      M(bottom_nodes, 2) = nan;
      M(bottom_nodes, 3) = 0;
   end
   M(side_nodes, :) = theta_dense(G, side_nodes, bcond(:, 1), ...
                                  bcond(:, 2), full_u, nan);
   M(bottom_edge_nodes, :) = theta_dense(G, bottom_edge_nodes, ...
                                         bcond(:, 1), bcond(:, 2), ...
                                         full_u, 0);
end

function U = theta_dense(G, nodes, prior, sigma, u, zdisp)
   if isempty(nodes)
      U = zeros(0, G.griddim);
      return
   end
   ipol = @(t, vmin, vmax) vmin * (1-t) + vmax * t;
   varval = @(ix) prior(ix) + ipol(u(ix), 0, sigma(ix));
   theta = varval(1);
   e1 = varval(2);
   e2 = varval(3);
   de1dz = varval(4);
   de2dz = varval(5);
   R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
   M0 = R * diag([e1, e2]) * R';
   M1 = R * diag([de1dz, de2dz]) * R';

   xy = G.nodes.coords(nodes, 1:2);
   z = G.nodes.coords(nodes, 3);
   U = zeros(numel(nodes), G.griddim);
   U(:, 1:2) = xy * M0' + (xy * M1') .* z;
   U(:, 3) = zdisp .* (1 + z);
end
