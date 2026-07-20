%% Dump Phase 10 reference: co2lab-spillpoint structural trap analysis.
%
% Runs MRST's cell-based trapAnalysis on two caprock surfaces:
%   1. a synthetic sloping surface with two domes (a known cascade), and
%   2. a synthetic multi-dome surface (several traps, richer spill graph).
% For each we save the caprock geometry plus the trap labels, spill depths,
% catchments and structural trapping capacities.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add co2lab-common co2lab-spillpoint coarsegrid matlab_bgl

cases = {'two_domes', 'multi_dome'};
for k = 1 : numel(cases)
   name = cases{k};
   switch name
     case 'two_domes'
       [nx, ny] = deal(60, 30);
       [Lx, Ly] = deal(1000, 1000);
       zf = @(x, y) 1000 - 0.02 * x ...
            - 12 * exp(-(((x - 250) / 90).^2 + ((y - 500) / 200).^2)) ...
            - 12 * exp(-(((x - 700) / 90).^2 + ((y - 500) / 200).^2));
     case 'multi_dome'
       [nx, ny] = deal(50, 50);
       [Lx, Ly] = deal(1000, 1000);
       zf = @(x, y) 1000 - 0.015 * x - 0.01 * y ...
            - 15 * exp(-(((x - 250) / 80).^2 + ((y - 250) / 80).^2)) ...
            - 10 * exp(-(((x - 700) / 90).^2 + ((y - 300) / 90).^2)) ...
            - 18 * exp(-(((x - 500) / 70).^2 + ((y - 750) / 70).^2));
   end

   G = cartGrid([nx, ny, 1], [Lx, Ly, 20]);
   x = G.nodes.coords(:, 1);  y = G.nodes.coords(:, 2);
   G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + zf(x, y);
   G = computeGeometry(G);

   [Gt, G] = topSurfaceGrid(G);
   res = trapAnalysis(Gt, true);        % true = cell-based algorithm

   poro = repmat(0.2, Gt.cells.num, 1);
   vols = volumesOfTraps(Gt, res, [], 'poro', poro);

   z_ve         = Gt.cells.z;
   H_ve         = Gt.cells.H;
   area_ve      = Gt.cells.volumes;
   traps        = res.traps;
   trap_z       = res.trap_z;
   trap_regions = res.trap_regions;
   trap_adj     = full(res.trap_adj);
   ntraps       = max(res.traps);

   save(fullfile(thisDir, sprintf('ref_p10_%s.mat', name)), '-v7', ...
        'nx', 'ny', 'Lx', 'Ly', 'z_ve', 'H_ve', 'area_ve', ...
        'traps', 'trap_z', 'trap_regions', 'trap_adj', 'ntraps', 'vols');
   fprintf('%-11s: %d traps, capacities %s\n', name, ntraps, ...
           mat2str(round(vols)));
end
fprintf('Wrote Phase-10 trap references to %s\n', thisDir);
