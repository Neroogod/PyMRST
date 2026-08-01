%% Export MRST gravity state transitions for PyMRST validation.

thisDir = fileparts(mfilename('fullpath'));
gravity reset; g0 = gravity;
gravity x on; g1 = gravity;
gravity off y; g2 = gravity;
gravity yes; g3 = gravity;
gravity reset true z; g4 = gravity;
gravity reset on; gravity(2.5); g5 = gravity;
gravity([1, 2]); g6 = gravity;
gravity([1, 2, -3]); g7 = gravity;
gravity(false); gravity([4, 5]); g8 = gravity;
gravity(true); g9 = gravity;
gravity x on reset; g10 = gravity;
gravity reset;

save(fullfile(thisDir, 'ref_params_gravity.mat'), '-v7', ...
    'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10');
fprintf('Wrote gravity reference to %s\n', thisDir);
