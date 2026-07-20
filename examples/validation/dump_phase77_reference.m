%% Dump Phase 77 reference: vemmech calibration Gaussian objective.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p77_vemmech_objective.mat; then run validate_phase77.py.

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
[bcfun, num_bc_ctrls] = set_bcfun(G, binfo, bcond);

props.young = 1.05e9 * ones(G.cells.num, 1);
props.poisson = 0.23 * ones(G.cells.num, 1);
props.rho = 1010 * ones(G.cells.num, 1);
matcond = [1 0.20 0.30 0.10];
[mfuns, num_mctrls] = set_material_fun(G, props, matcond, num_bc_ctrls);
mat_facs = reshape(matcond(:, 2:end), [], 1);

u_init = zeros(num_bc_ctrls + sum(num_mctrls), 1);
u_sigma = ones(size(u_init));
u = linspace(0.2, 0.8, numel(u_init))';

bc_dense = dense_bcfun_reference(G, binfo, bcond, u(1:num_bc_ctrls));
dd_mat = zeros(G.nodes.num, G.griddim);
fixed = ~isnan(bc_dense);
dd_mat(fixed) = bc_dense(fixed);
dd = reshape(dd_mat', [], 1);

C = Enu2C(mfuns.efun(u), mfuns.nufun(u), G);
[~, op] = VEM_assemble(G, C);
stress = calculateStressVEM(G, dd_mat, op);
[Smax, Smin, theta] = Sigma2SmaxSminTheta(stress(:, 1), ...
                                          stress(:, 2), ...
                                          stress(:, 4));
Sv = stress(:, 3);

data.orientation = [1, theta(1) + 0.03, 0.12];
data.Smax = [1, 1.05 * Smax(1), 0.08];
data.Smin = [1, 0.94 * Smin(1), 0.09];
data.Sv = [1, 0.97 * Sv(1), 0.07];
model_sigma = [0.01, 2e5, 2.5e5, 3e5];
scaling = 2.5;

[val, terms, simvals, datapts, all_simvals] = ...
    dense_gaussian_objfun_reference(u, stress, data, model_sigma, ...
                                    num_bc_ctrls, mat_facs, u_init, ...
                                    u_sigma, scaling, false);
[match_val, match_terms] = ...
    dense_gaussian_objfun_reference(u, stress, data, model_sigma, ...
                                    num_bc_ctrls, mat_facs, u_init, ...
                                    u_sigma, scaling, true);

save(fullfile(thisDir, 'ref_p77_vemmech_objective.mat'), '-v7', ...
     'bcond', 'num_bc_ctrls', 'num_mctrls', 'props', 'matcond', ...
     'mat_facs', 'u_init', 'u_sigma', 'u', 'dd', 'stress', 'data', ...
     'model_sigma', 'scaling', 'val', 'match_val', 'terms', ...
     'match_terms', 'simvals', 'datapts', 'all_simvals');

fprintf('Wrote ref_p77_vemmech_objective.mat to %s\n', thisDir);

function [val, terms, simvals, datapts, all_simvals] = ...
    dense_gaussian_objfun_reference(u, stress, data, model_sigma, ...
                                    num_bc_ctrls, mat_facs, u_prior, ...
                                    u_sigma, scaling, ignore_prior)
   orientation = data.orientation;
   Smax_data = data.Smax;
   Smin_data = data.Smin;
   Sv_data = data.Sv;
   d_cell = [orientation(:, 1); Smax_data(:, 1); ...
             Smin_data(:, 1); Sv_data(:, 1)];
   d_obs = [orientation(:, 2); Smax_data(:, 2); ...
            Smin_data(:, 2); Sv_data(:, 2)];
   d_sigma = [orientation(:, 3); Smax_data(:, 3); ...
              Smin_data(:, 3); Sv_data(:, 3)];
   d_type = [ones(size(orientation, 1), 1); ...
             2 * ones(size(Smax_data, 1), 1); ...
             3 * ones(size(Smin_data, 1), 1); ...
             4 * ones(size(Sv_data, 1), 1)];

   Sv = stress(:, 3);
   [Smax, Smin, theta] = Sigma2SmaxSminTheta(stress(:, 1), ...
                                             stress(:, 2), ...
                                             stress(:, 4));
   theta_ixs = find(d_type == 1); theta_cells = d_cell(theta_ixs);
   Smax_ixs  = find(d_type == 2); Smax_cells  = d_cell(Smax_ixs);
   Smin_ixs  = find(d_type == 3); Smin_cells  = d_cell(Smin_ixs);
   Sv_ixs    = find(d_type == 4); Sv_cells    = d_cell(Sv_ixs);

   threshold_theta = 5e-2;
   relDs = @(cells, th) abs(Smax(cells) - Smin(cells)) ./ ...
                         (th * abs(Smax(cells)));
   theta_relDs = relDs(theta_cells, threshold_theta);
   thetaIso = theta_relDs < 1;
   theta_sigma = sqrt(d_sigma(theta_ixs).^2 + model_sigma(1).^2);
   r_theta = dense_thetadiff(d_obs(theta_ixs), theta(theta_cells)) ./ ...
             theta_sigma;
   r_theta_iso = pi/2 * ones(sum(thetaIso), 1) ./ theta_sigma(thetaIso);
   onevec = ones(sum(thetaIso), 1);
   scaling_fun = 0.5 * (onevec - cos(theta_relDs(thetaIso) * pi));
   r_theta(thetaIso) = scaling_fun .* r_theta(thetaIso) + ...
                       (onevec - scaling_fun) .* r_theta_iso;

   Smax_sigma = sqrt((d_sigma(Smax_ixs) .* d_obs(Smax_ixs)).^2 + ...
                     model_sigma(2).^2);
   r_Smax = (d_obs(Smax_ixs) - Smax(Smax_cells)) ./ Smax_sigma;
   Smin_sigma = sqrt((d_sigma(Smin_ixs) .* d_obs(Smin_ixs)).^2 + ...
                     model_sigma(3).^2);
   r_Smin = (d_obs(Smin_ixs) - Smin(Smin_cells)) ./ Smin_sigma;
   Sv_sigma = sqrt((d_sigma(Sv_ixs) .* d_obs(Sv_ixs)).^2 + ...
                   model_sigma(4).^2);
   r_Sv = (d_obs(Sv_ixs) - Sv(Sv_cells)) ./ Sv_sigma;

   match_term = sum(r_theta.^2) + sum(r_Smin.^2) + ...
                sum(r_Smax.^2) + sum(r_Sv.^2);
   num_terms = numel(d_obs);

   r_prior = (u - u_prior(:)) ./ u_sigma;
   mat_ind = false(numel(u), 1);
   mat_ind(num_bc_ctrls+1:end) = true;
   if any(mat_ind)
      r_prior(mat_ind) = log(1 + r_prior(mat_ind) .* mat_facs) ./ ...
          sqrt(log(1 + mat_facs.^2));
   end
   if ~ignore_prior
      match_term = match_term + sum(r_prior.^2);
   end
   val = match_term / (num_terms * scaling);
   terms.r_theta = r_theta;
   terms.r_Smax = r_Smax;
   terms.r_Smin = r_Smin;
   terms.r_Sv = r_Sv;
   terms.r_prior = r_prior;
   simvals.Smax = Smax(Smax_cells);
   simvals.Smin = Smin(Smin_cells);
   simvals.Sv = Sv(Sv_cells);
   simvals.theta = theta(theta_cells);
   datapts.theta = [d_cell(theta_ixs), d_obs(theta_ixs), theta_sigma];
   datapts.Smax = [d_cell(Smax_ixs), d_obs(Smax_ixs), Smax_sigma];
   datapts.Smin = [Smin_cells(:), d_obs(Smin_ixs), Smin_sigma];
   datapts.Sv = [Sv_cells(:), d_obs(Sv_ixs), Sv_sigma];
   all_simvals.theta = theta;
   all_simvals.Smax = Smax;
   all_simvals.Smin = Smin;
   all_simvals.Sv = Sv;
end

function dth = dense_thetadiff(theta1, theta2)
   dth = abs(theta1 - theta2);
   while any(dth > pi)
      ixs = dth > pi;
      dth(ixs) = dth(ixs) - pi * ones(sum(ixs), 1);
   end
   ixs = dth > pi/2;
   dth(ixs) = pi * ones(sum(ixs), 1) - dth(ixs);
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
