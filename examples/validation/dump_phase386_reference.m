%% Dump Phase 386 reference: permeabilityConverter (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p386_permconv.mat.
%
% Pins the PyMRST `permeability_converter` port against MRST's
% `permeabilityConverter` on a *tilted* corner-point deck (sloped layers, so the
% cell axes are not the global axes and the rotation produces off-diagonal
% terms). Diagonal PERMX/Y/Z are rotated to the global frame in the 'cell'
% system. The input deck is dumped alongside the output tensor so PyMRST rebuilds
% the identical input and compares all six components.

mrstModule add deckformat

thisDir = fileparts(mfilename('fullpath'));

g = simpleGrdecl([3, 2, 2], @(x) 0.15 * sin(2*pi*x));   % tilted / wavy layers
nc = prod(g.cartDims);
g.PERMX = (100 + (1:nc))';
g.PERMY = (50  + (1:nc))';
g.PERMZ = (10  + 0.5*(1:nc))';

in_cartDims = g.cartDims(:);
in_COORD = g.COORD(:); in_ZCORN = g.ZCORN(:);
in_PERMX = g.PERMX(:); in_PERMY = g.PERMY(:); in_PERMZ = g.PERMZ(:);

g2 = permeabilityConverter(g, 'K_system', 'cell');

out_PERMX  = g2.PERMX(:);   out_PERMY  = g2.PERMY(:);   out_PERMZ  = g2.PERMZ(:);
out_PERMXY = g2.PERMXY(:);  out_PERMYZ = g2.PERMYZ(:);  out_PERMZX = g2.PERMZX(:);

save(fullfile(thisDir, 'ref_p386_permconv.mat'), '-v7', ...
     'in_cartDims', 'in_COORD', 'in_ZCORN', 'in_PERMX', 'in_PERMY', 'in_PERMZ', ...
     'out_PERMX', 'out_PERMY', 'out_PERMZ', 'out_PERMXY', 'out_PERMYZ', 'out_PERMZX');

fprintf('Wrote ref_p386_permconv.mat to %s\n', thisDir);
fprintf('  %d cells; PERMXY range [%.4f, %.4f]\n', ...
        nc, min(out_PERMXY), max(out_PERMXY));
