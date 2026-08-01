%% Export makeRock and permTensor layout references.

thisDir = fileparts(mfilename('fullpath'));

G1 = cartGrid(3);
r1 = makeRock(G1, 2, 0.2);
[K1, row1, col1] = permTensor(r1, 1);

G2 = cartGrid([2, 2]);
r2s = makeRock(G2, 2, 0.2, 'ntg', 0.8);
[K2s, row2, col2] = permTensor(r2s, 2);
r2d = makeRock(G2, [2, 3], [0.1; 0.2; 0.3; 0.4]);
K2d = permTensor(r2d, 2);
r2f = makeRock(G2, [2, 0.5, 3], 0.25);
K2f = permTensor(r2f, 2);

G3 = cartGrid([2, 1, 2]);
r3s = makeRock(G3, 2, 0.2);
[K3s, row3, col3] = permTensor(r3s, 3);
r3d = makeRock(G3, [2, 3, 4], 0.2);
K3d = permTensor(r3d, 3);
r3f = makeRock(G3, [2, 0.1, 0.2, 3, 0.3, 4], 0.2);
K3f = permTensor(r3f, 3);

perm2d = r2d.perm; poro2d = r2d.poro;
perm3f = r3f.perm; ntg2s = r2s.ntg;
save(fullfile(thisDir, 'ref_params_rock_tensor.mat'), '-v7', ...
    'K1', 'row1', 'col1', 'K2s', 'row2', 'col2', 'K2d', 'K2f', ...
    'K3s', 'row3', 'col3', 'K3d', 'K3f', ...
    'perm2d', 'poro2d', 'perm3f', 'ntg2s');
fprintf('Wrote makeRock/permTensor reference to %s\n', thisDir);
