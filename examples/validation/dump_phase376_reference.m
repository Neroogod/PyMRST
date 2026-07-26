%% Dump Phase 376 reference: accumulateCartesianMultipliers (Phase 25), MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p376_multipliers.mat.
%
% Pins the PyMRST `accumulate_cartesian_multipliers` port against MRST's
% `accumulateCartesianMultipliers` on a 3x2x2 grid with deterministic per-cell
% MULTX/MULTY/MULTZ arrays. The resulting per-face multiplier vector is compared
% number for number (cartGrid face ordering matches between the two codes).

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 2, 2], [3, 2, 2]));
nc = G.cells.num;

mult.X = 1 + 0.1 * (1:nc)';
mult.Y = 2 - 0.05 * (1:nc)';
mult.Z = 0.5 + 0.02 * (1:nc)';

facemult = accumulateCartesianMultipliers(G, mult);

multX = mult.X; multY = mult.Y; multZ = mult.Z;

save(fullfile(thisDir, 'ref_p376_multipliers.mat'), '-v7', ...
     'facemult', 'multX', 'multY', 'multZ');

fprintf('Wrote ref_p376_multipliers.mat to %s\n', thisDir);
fprintf('  %d cells -> %d face multipliers, range [%.4f, %.4f]\n', ...
        nc, numel(facemult), min(facemult), max(facemult));
