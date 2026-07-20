%% Dump Phase 70 reference: VEM_linElastTEDFM solve wrapper.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p70_vem_tedfm_solve.mat into this folder; then run validate_phase70.py.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

E = 1e9;
nu = 0.25;
nx = 3;
ny = 2;
L = [3, 2];

G = computeGeometry(cartGrid([nx, ny], L));
G = createAugmentedGrid(G);
Gwrap = G;
Gwrap.Matrix = G;
C = Enu2C(E * ones(G.cells.num, 1), nu * ones(G.cells.num, 1), G);

coords = G.nodes.coords;
bottom_nodes = find(coords(:, 2) < 1e-9);
left_nodes = find(coords(:, 1) < 1e-9);
disp_nodes = [bottom_nodes; left_nodes];
disp_mask = [repmat([false, true], numel(bottom_nodes), 1); ...
             repmat([true, false], numel(left_nodes), 1)];
disp_uu = zeros(numel(disp_nodes), G.griddim);

bd = any(G.faces.neighbors == 0, 2);
top_faces = find(bd & G.faces.centroids(:, 2) > L(2) - 1e-9);
force = repmat([0, -1e6], numel(top_faces), 1);

el_bc = struct();
el_bc.disp_bc = struct('nodes', disp_nodes, ...
                       'uu', disp_uu, ...
                       'mask', disp_mask);
el_bc.force_bc = struct('faces', top_faces, 'force', force);

load = @(x) zeros(size(x));

[uu, extra] = VEM_linElastTEDFM(Gwrap, C, el_bc, load);
[stress, strain] = calculateStressVEM(G, uu, extra);

S = extra.S;
A = extra.A;
rhs = extra.rhs;
isdirdofs = extra.disc.isdirdofs;
V_dir = extra.disc.V_dir;
ovol_div = extra.disc.ovol_div;
gradP = extra.disc.gradP;
div = extra.disc.div;
divrhs = extra.disc.divrhs;

save(fullfile(thisDir, 'ref_p70_vem_tedfm_solve.mat'), '-v7', ...
     'E', 'nu', 'nx', 'ny', 'L', ...
     'bottom_nodes', 'left_nodes', 'top_faces', ...
     'C', 'uu', 'stress', 'strain', 'S', 'A', 'rhs', 'isdirdofs', 'V_dir', ...
     'ovol_div', 'gradP', 'div', 'divrhs');

fprintf('Wrote ref_p70_vem_tedfm_solve.mat to %s\n', thisDir);
