%% Dump Phase 419 reference: DG element cubatures, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p419_dg_element_cubature.mat.
%
% Pins PyMRST's Triangle/Line/Tetrahedron cubatures against MRST's classes:
% dumps the per-element cubature points, weights and positions on a 2-D grid
% (TriangleCubature over cells, LineCubature over faces) and a 3-D grid
% (TetrahedronCubature over cells, TriangleCubature over faces).

mrstModule add dg vem vemmech

thisDir = fileparts(mfilename('fullpath'));
out = struct();

% --- 2-D grid: cells (triangle) and faces (line).
G2 = computeGeometry(cartGrid([3, 2], [3, 2]));
ct = TriangleCubature(G2, 3);
out.tri2_x = ct.points; out.tri2_w = ct.weights; out.tri2_pos = ct.pos;
cl = LineCubature(G2, 3);
out.line2_x = cl.points; out.line2_w = cl.weights; out.line2_pos = cl.pos;

% --- 3-D grid: cells (tetrahedron) and faces (triangle).
G3 = computeVEMGeometry(cartGrid([2, 2, 2], [2, 2, 2]));
ctet = TetrahedronCubature(G3, 3);
out.tet3_x = ctet.points; out.tet3_w = ctet.weights; out.tet3_pos = ctet.pos;
ctf = TriangleCubature(G3, 3);
out.trif3_x = ctf.points; out.trif3_w = ctf.weights; out.trif3_pos = ctf.pos;

save(fullfile(thisDir, 'ref_p419_dg_element_cubature.mat'), '-v7', '-struct', 'out');
fprintf('Wrote ref_p419_dg_element_cubature.mat\n');
