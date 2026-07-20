%% Dump Phase 71 reference: vemmech grid utilities.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p71_vemmech_grid_utils.mat; then run validate_phase71.py.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

nx = 2; ny = 2; nz = 2;
coord = [];
for j = 0:ny
    for i = 0:nx
        coord = [coord; i; j; 0; i + 0.1; j + 0.2; nz]; %#ok<AGROW>
    end
end
zcorn = zeros(2*nx, 2*ny, 2*nz);
for k = 1:nz
    zcorn(:, :, 2*k - 1) = k - 1;
    zcorn(:, :, 2*k) = k;
end
grdecl = struct('cartDims', [nx, ny, nz], ...
                'COORD', coord, ...
                'ZCORN', zcorn(:));

v_all = verticalGrdecl(grdecl, 'method', 'all');
v_sides = verticalGrdecl(grdecl, 'method', 'sides');

box = [1 2; 3 4; 5 6];
pad_z = padGrdecl(grdecl, [false false true], box);
pad_xz = padGrdecl(grdecl, [true false true], box);
pad_yz = padGrdecl(grdecl, [false true true], box);
pad_xyz = padGrdecl(grdecl, [true true true], box);
pad_xz_rel = padGrdecl(grdecl, [true false true], box, 'relative', true);

G = computeGeometry(cartGrid([2, 2, 2], [2, 4, 6]));
Gf = flipGrid(G);
flip_coords = Gf.nodes.coords;
flip_tags = Gf.cells.faces(:, 2);

save(fullfile(thisDir, 'ref_p71_vemmech_grid_utils.mat'), '-v7', ...
     'grdecl', 'v_all', 'v_sides', 'pad_z', 'pad_xz', 'pad_yz', ...
     'pad_xyz', 'pad_xz_rel', 'flip_coords', 'flip_tags');

fprintf('Wrote ref_p71_vemmech_grid_utils.mat to %s\n', thisDir);
