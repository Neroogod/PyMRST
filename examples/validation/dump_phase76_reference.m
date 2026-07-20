%% Dump Phase 76 reference: vemmech calibration calculate_stress.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p76_vemmech_calculate_stress.mat; then run validate_phase76.py.

mrstModule add ad-core vemmech

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

E = 1.05e9 * ones(G.cells.num, 1);
nu = 0.23 * ones(G.cells.num, 1);
C = Enu2C(E, nu, G);
[~, op] = VEM_assemble(G, C);

x = G.nodes.coords(:, 1);
y = G.nodes.coords(:, 2);
z = G.nodes.coords(:, 3);
dd_mat = zeros(G.nodes.num, G.griddim);
bc_dense = dense_bcfun_reference(G, binfo, bcond, u);
fixed = ~isnan(bc_dense);
dd_full_mat = dd_mat;
dd_full_mat(fixed) = bc_dense(fixed);
dd = reshape(dd_mat', [], 1);

stress = calculateStressVEM(G, dd_full_mat, op);
stress_vec = reshape(stress', [], 1);

save(fullfile(thisDir, 'ref_p76_vemmech_calculate_stress.mat'), '-v7', ...
     'bcond', 'num_bc', 'u', 'E', 'nu', 'dd', 'stress', 'stress_vec');

fprintf('Wrote ref_p76_vemmech_calculate_stress.mat to %s\n', thisDir);

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
