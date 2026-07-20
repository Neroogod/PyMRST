%% Dump Phase 74 reference: vemmech calibration helpers.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p74_vemmech_calibration.mat; then run validate_phase74.py.

mrstModule add ad-core vemmech
gravity reset on

thisDir = fileparts(mfilename('fullpath'));

o_xx = [-3; -1; -2; -4; -1.5];
o_yy = [-1; -3; -2; -1; -2.5];
o_xy = [0; 0; 1; -0.5; 0.25];
[Smax, Smin, theta] = Sigma2SmaxSminTheta(o_xx, o_yy, o_xy);

G = computeGeometry(cartGrid([2, 2, 3], [2, 2, 3]));
G = createAugmentedGrid(G);

binfo = identify_boundary_elements(G);
side_faces = binfo.side_faces;
side_nodes = binfo.side_nodes;

props.young = 1e9 + (1:G.cells.num)' * 1e7;
props.poisson = 0.20 + (1:G.cells.num)' * 0.002;
props.rho = 1000 + (1:G.cells.num)' * 3;
props_young = props.young;
props_poisson = props.poisson;
props_rho = props.rho;

[funs0, num0] = set_material_fun(G, props, [], 0);
E0 = funs0.efun([]);
nu0 = funs0.nufun([]);
L0 = full(funs0.loadfun([]).asMatrix({{'cn'}, {'d'}}, true, ...
                                      [numel(G.cells.nodes), G.griddim]));

matcond = [1 0.20 0.00 0.10; ...
           2 0.00 0.30 0.00; ...
           3 0.40 0.50 0.20];
[funs, num] = set_material_fun(G, props, matcond, 0);
u = linspace(0.1, 0.9, sum(num))';
E = funs.efun(u);
nu = funs.nufun(u);
L = full(funs.loadfun(u).asMatrix({{'cn'}, {'d'}}, true, ...
                                  [numel(G.cells.nodes), G.griddim]));

save(fullfile(thisDir, 'ref_p74_vemmech_calibration.mat'), '-v7', ...
     'o_xx', 'o_yy', 'o_xy', 'Smax', 'Smin', 'theta', ...
     'side_faces', 'side_nodes', ...
     'props_young', 'props_poisson', 'props_rho', ...
     'num0', 'E0', 'nu0', 'L0', ...
     'matcond', 'u', 'num', 'E', 'nu', 'L');

fprintf('Wrote ref_p74_vemmech_calibration.mat to %s\n', thisDir);
