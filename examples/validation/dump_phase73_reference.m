%% Dump Phase 73 reference: vemmech AD tensor utilities.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p73_vemmech_ad_utils.mat; then run validate_phase73.py.

mrstModule add ad-core vemmech

thisDir = fileparts(mfilename('fullpath'));

t = SparseMultiArray([2; 3], [1 2; 2 1], {'i', 'j'});
t_mat = full(t.asMatrix({{'i'}, {'j'}}, true, [2 2]));
t_vec_ji = full(t.asVector({'j', 'i'}, [2 2]));

kron_ref = kron_ADI([2; 3], [5; 7; 11]);

G = computeGeometry(cartGrid([1, 1, 1], [1, 2, 3]));
fnt = face_node_tensor(G, 'f', 'n');
fnt_mat = full(fnt.asMatrix({{'f'}, {'n'}}, true, [G.faces.num, G.nodes.num]));
vals = (1:numel(G.faces.nodes))';
fnt_val = face_node_tensor(G, 'f', 'n', 'values', vals);
fnt_val_mat = full(fnt_val.asMatrix({{'f'}, {'n'}}, true, ...
                                    [G.faces.num, G.nodes.num]));
fnt_b = face_node_tensor(G, 'f', 'n', 'boundary_only', true);
fnt_b_mat = full(fnt_b.asMatrix({{'f'}, {'n'}}, true, ...
                                [G.faces.num, G.nodes.num]));

E = 1e9 * ones(G.cells.num, 1);
nu = 0.25 * ones(G.cells.num, 1);
C = Enu2C(E, nu, G);

save(fullfile(thisDir, 'ref_p73_vemmech_ad_utils.mat'), '-v7', ...
     't_mat', 't_vec_ji', 'kron_ref', ...
     'fnt_mat', 'fnt_val_mat', 'fnt_b_mat', 'C');

fprintf('Wrote ref_p73_vemmech_ad_utils.mat to %s\n', thisDir);
