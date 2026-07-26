%% Dump Phase 378 reference: repairNormals (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p378_repair.mat.
%
% Pins the PyMRST `repair_normals` port against MRST's `repairNormals` on an
% irregular two-cell 2D grid (non-axis-aligned quads). Cell areas ("volumes")
% are compared per cell (cell order is the tessellation order in both codes) and
% the face lengths ("areas") as a sorted set (face renumbering may differ).

thisDir = fileparts(mfilename('fullpath'));

p = [0 0; 2 0; 2 1; 0 1.5; 4 0.3; 4 1.2];
t = [1 2 3 4; 2 5 6 3];
G = tessellationGrid(p, t);
G = repairNormals(G);

cell_volumes = G.cells.volumes(:);
face_areas   = sort(G.faces.areas(:));

save(fullfile(thisDir, 'ref_p378_repair.mat'), '-v7', ...
     'cell_volumes', 'face_areas', 'p', 't');

fprintf('Wrote ref_p378_repair.mat to %s\n', thisDir);
fprintf('  cell areas = [%.4f %.4f], total edge length = %.4f\n', ...
        cell_volumes, sum(face_areas));
