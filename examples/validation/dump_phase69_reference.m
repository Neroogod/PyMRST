%% Dump Phase 69 reference: TEDFM VEM wrappers.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p69_vem_tedfm.mat into this folder; then run validate_phase69.py.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

E = 1e9;
nu = 0.25;

G2 = computeGeometry(cartGrid([3, 2], [3, 2]));
G2 = createAugmentedGrid(G2);
G2w = struct('Matrix', G2, 'griddim', G2.griddim);
C2 = Enu2C(E * ones(G2.cells.num, 1), nu * ones(G2.cells.num, 1), G2);
[qf2, qfvol2] = calculateQFTEDFM(G2w);
D2 = C2DTEDFM(C2, G2w);
S2 = VEM_assembleTEDFM(G2w, C2);
G2_cell_nodes = G2.cells.nodes;
G2_cell_node_pos = G2.cells.nodePos;

G3 = computeGeometry(cartGrid([2, 2, 2], [2, 2, 2]));
G3 = createAugmentedGrid(G3);
G3w = struct('Matrix', G3, 'griddim', G3.griddim);
C3 = Enu2C(E * ones(G3.cells.num, 1), nu * ones(G3.cells.num, 1), G3);
[qc3, qf3, qcvol3] = calculateQCTEDFM(G3w);
D3 = C2DTEDFM(C3, G3w);
div3 = VEM3D_divTEDFM(G3w);
vdiv3 = VEM_divTEDFM(G3w);
S3 = VEM_assembleTEDFM(G3w, C3);

save(fullfile(thisDir, 'ref_p69_vem_tedfm.mat'), '-v7', ...
     'E', 'nu', ...
     'qf2', 'qfvol2', 'C2', 'D2', 'S2', ...
     'G2_cell_nodes', 'G2_cell_node_pos', ...
     'qc3', 'qf3', 'qcvol3', 'C3', 'D3', 'div3', 'vdiv3', 'S3');

fprintf('Wrote ref_p69_vem_tedfm.mat to %s\n', thisDir);
