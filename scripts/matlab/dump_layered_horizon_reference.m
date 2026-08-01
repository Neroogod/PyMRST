% Reference for validate_make_layered_horizon_grid.py
out = struct();
G2D = computeGeometry(cartGrid([4,4],[1,1]));

flat0  = mkc(9, @(x,y) zeros(size(x)));
flat5  = mkc(9, @(x,y) 5*ones(size(x)));
flat9  = mkc(9, @(x,y) 9*ones(size(x)));
flat10 = mkc(9, @(x,y) 10*ones(size(x)));
slope  = mkc(9, @(x,y) 5*x);

cases = { {{flat0, flat10}, {'layers', 3}}, ...
          {{flat0, flat10}, {}}, ...
          {{flat0, flat5, flat9}, {'layers', [2 3]}}, ...
          {{slope, flat10}, {}}, ...
          {{flat0, slope}, {}} };
names = {'layers3','one','multi','slope','wedge'};

for k = 1:numel(cases)
   [G, cellmap] = makeLayeredHorizonGrid(G2D, cases{k}{1}, cases{k}{2}{:});
   G = computeGeometry(G);
   r = struct();
   r.numCells = G.cells.num;
   r.numNodes = G.nodes.num;
   r.totVol   = sum(G.cells.volumes);
   r.zsorted  = sort(G.nodes.coords(:,3));
   r.volsorted= sort(G.cells.volumes);
   r.cellmap  = cellmap(:);
   r.layer    = G.cells.layer(:);
   out.(names{k}) = r;
end

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_layered_horizon.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_layered_horizon.mat\n');

function h = mkc(n, f)
  [X, Y] = meshgrid(linspace(-0.1, 1.1, n), linspace(-0.1, 1.1, n));
  h = struct('x', X, 'y', Y, 'z', f(X, Y));
end
