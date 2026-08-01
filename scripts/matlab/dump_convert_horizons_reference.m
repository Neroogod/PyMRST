% Reference for validate_convert_horizons_to_grid.py
out = struct();

flat0  = mkh(8, 0, 1, 0, 1, @(x,y) zeros(size(x)));
flat5  = mkh(8, 0, 1, 0, 1, @(x,y) 5*ones(size(x)));
flat10 = mkh(8, 0, 1, 0, 1, @(x,y) 10*ones(size(x)));
slope  = mkh(8, 0, 1, 0, 1, @(x,y) 5*x);
offset = mkh(8, 0.5, 1.5, 0, 1, @(x,y) 10*ones(size(x)));

cases = { {{flat0, flat10}, {}}, ...
          {{flat0, flat10}, {'layers', 3}}, ...
          {{slope, flat10}, {}}, ...
          {{flat0, flat5, flat10}, {'layers', [2 3]}}, ...
          {{flat0, flat10}, {'dims', [20 15]}}, ...
          {{flat0, offset}, {}} };
names = {'box','layers3','slope','multi','dims','offset'};

for k = 1:numel(cases)
   g = convertHorizonsToGrid(cases{k}{1}, cases{k}{2}{:});
   r = struct();
   r.cartDims = g.cartDims;
   r.COORD  = g.COORD;
   r.ZCORN  = g.ZCORN;
   r.ACTNUM = double(g.ACTNUM);
   out.(names{k}) = r;
end

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_convert_horizons.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_convert_horizons.mat\n');

function h = mkh(n, x0, x1, y0, y1, f)
  [X, Y] = ndgrid(linspace(x0, x1, n), linspace(y0, y1, n));
  h = struct('x', X, 'y', Y, 'z', f(X, Y));
end
