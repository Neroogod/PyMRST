%% Dump Phase 7 reference: vemmech VEM_linElast displacement fields.
%
% Two cases on a 3x4x5 box: a regular Cartesian grid and a node-twisted
% version, both with heterogeneous E/nu, bottom nodes fully clamped,
% side rollers via masks, a traction on the top faces, and a body load.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add vemmech

for icase = 1 : 2
   G = cartGrid([3, 4, 5], [3, 4, 5]);
   if icase == 2
      pts = G.nodes.coords;
      interior = all(pts(:, 1:2) > 1e-9, 2) & ...
                 pts(:, 1) < 3 - 1e-9 & pts(:, 2) < 4 - 1e-9;
      pts(interior, 1) = pts(interior, 1) + 0.15 * sin(pts(interior, 3));
      pts(interior, 2) = pts(interior, 2) + 0.10 * cos(pts(interior, 1));
      G.nodes.coords = pts;
   end
   G = computeGeometry(G);
   G = createAugmentedGrid(G);
   nc = G.cells.num;

   E  = 1e9 * (1 + 0.5 * mod((0 : nc - 1)', 3));
   nu = 0.20 + 0.05 * mod((0 : nc - 1)', 4);
   C  = Enu2C(E, nu, G);

   pts = G.nodes.coords;
   bn  = find(pts(:, 3) > 5 - 1e-9);              % bottom nodes (z = 5)
   bd  = any(G.faces.neighbors == 0, 2);
   topf = find(bd & G.faces.centroids(:, 3) < 1e-9);

   el_bc = struct('disp_bc', struct('nodes', bn, ...
                                    'uu', zeros(numel(bn), 3), ...
                                    'mask', true(numel(bn), 3)), ...
                  'force_bc', struct('faces', topf, ...
                                     'force', repmat([1e5, -5e4, 2e6], ...
                                                     numel(topf), 1)));
   load = @(x) repmat([100, 200, -3000], size(x, 1), 1);

   uu = VEM_linElast(G, C, el_bc, load);                          %#ok

   coords = G.nodes.coords;
   save(fullfile(thisDir, sprintf('ref_p7_vem%d.mat', icase)), ...
        '-v7', 'uu', 'coords', 'E', 'nu');
end
fprintf('Wrote 2 Phase-7 reference files to %s\n', thisDir);
