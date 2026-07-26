%% Dump Phase 374 reference: refineGrdecl (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p374_refine.mat.
%
% Pins the PyMRST `refine_grdecl`/`grdecl_xyz` ports against MRST's `refineGrdecl`
% on a non-trivial (wavy) corner-point deck refined by [2 2 2]. The input deck is
% dumped alongside the refined output, so PyMRST refines the identical input and
% compares COORD, ZCORN, cartDims and the inherited PERMX number for number.

thisDir = fileparts(mfilename('fullpath'));

g_in = simpleGrdecl([3, 2, 2], @(x) 0.05 * sin(2*pi*x));   % wavy pillars
g_in.PERMX = (1:prod(g_in.cartDims))';                     % a cell field

dim = [2, 2, 2];
g_out = refineGrdecl(g_in, dim);

in_cartDims  = g_in.cartDims(:);
in_COORD     = g_in.COORD(:);
in_ZCORN     = g_in.ZCORN(:);
in_PERMX     = g_in.PERMX(:);

out_cartDims = g_out.cartDims(:);
out_COORD    = g_out.COORD(:);
out_ZCORN    = g_out.ZCORN(:);
out_PERMX    = g_out.PERMX(:);

save(fullfile(thisDir, 'ref_p374_refine.mat'), '-v7', ...
     'in_cartDims', 'in_COORD', 'in_ZCORN', 'in_PERMX', 'dim', ...
     'out_cartDims', 'out_COORD', 'out_ZCORN', 'out_PERMX');

fprintf('Wrote ref_p374_refine.mat to %s\n', thisDir);
fprintf('  cartDims [%d %d %d] -> [%d %d %d]\n', in_cartDims, out_cartDims);
