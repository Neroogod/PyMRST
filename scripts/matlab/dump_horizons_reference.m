% Reference for validate_sort_horizons.py and validate_refine_grdecl_layers.py
out = struct();

% --- sortHorizons
[X, Y] = meshgrid(0:2, 0:2);
h = {struct('x', X, 'y', Y, 'z', 30*ones(3)), ...
     struct('x', X, 'y', Y, 'z', 10*ones(3)), ...
     struct('x', X, 'y', Y, 'z', 20*ones(3))};
[hs, ix] = sortHorizons(h);
out.sorted_ix = ix;
out.sorted_z  = cellfun(@(s) s.z(1,1), hs);

% A tilted surface: min decides, not the mean.
tilt = struct('x', X, 'y', Y, 'z', repmat([0 100 100], 3, 1));
flat = struct('x', X, 'y', Y, 'z', 50*ones(3));
[~, ix2] = sortHorizons({flat, tilt});
out.tilt_ix = ix2;

% --- refineGrdeclLayers
nx = 3; ny = 3; nz = 4;
xyz = zeros(6, nx+1, ny+1);
for i = 1:nx+1
  for j = 1:ny+1
    xyz(:, i, j) = [i-1; j-1; 0; i-1; j-1; nz];
  end
end
zc = zeros(2*nx, 2*ny, 2*nz);
for k = 1:nz
  zc(:, :, 2*k-1) = k-1;
  zc(:, :, 2*k)   = k;
end
grdecl = struct('cartDims', [nx ny nz], 'COORD', xyz(:), 'ZCORN', zc(:));

cases = {[2 3], [3 3], [1 1], [1 4]};
refs  = {3, 5, 2, 2};
names = {'band23','single3','first','all'};
for k = 1:numel(cases)
   ng = refineGrdeclLayers(grdecl, cases{k}, refs{k});
   r = struct();
   r.cartDims = ng.cartDims;
   [~, z] = grdeclXYZ(ng);
   r.zcorn = z;
   out.(names{k}) = r;
end

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_horizons.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_horizons.mat\n');
