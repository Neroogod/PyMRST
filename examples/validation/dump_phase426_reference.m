%% Dump Phase 426 reference: computeCpGeometry, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p426_cp_geometry.mat.

mrstModule add deckformat ad-core

thisDir = fileparts(mfilename('fullpath'));

grdecl = simpleGrdecl([3, 2, 2], @(x) 0.12*sin(2*pi*x));
grdecl.ACTNUM = ones(prod(grdecl.cartDims), 1);
grdecl.ACTNUM(5) = 0;
G = processGRDECL(grdecl);

[out_active_cc, out_active_fc, out_dims] = computeCpGeometry(G, grdecl);
[out_all_cc, out_all_fc, out_all_dims] = ...
    computeCpGeometry(G, grdecl, 'on_all_cpc', true);

in_cartDims = grdecl.cartDims(:);
in_COORD = grdecl.COORD(:);
in_ZCORN = grdecl.ZCORN(:);
in_ACTNUM = grdecl.ACTNUM(:);

save(fullfile(thisDir, 'ref_p426_cp_geometry.mat'), '-v7', ...
     'in_cartDims', 'in_COORD', 'in_ZCORN', 'in_ACTNUM', ...
     'out_active_cc', 'out_active_fc', 'out_dims', ...
     'out_all_cc', 'out_all_fc', 'out_all_dims');

fprintf('Wrote ref_p426_cp_geometry.mat to %s\n', thisDir);
fprintf('  %d active of %d Cartesian cells, %d half-faces\n', ...
        G.cells.num, prod(grdecl.cartDims), size(out_active_fc, 1));
