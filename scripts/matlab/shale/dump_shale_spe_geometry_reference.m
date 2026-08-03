function dump_shale_spe_geometry_reference(output_path)
% Direct MRST fixtures for deterministic SPE-201243-PA geometry owners.
if nargin ~= 1, error('Expected explicit output_path.'); end
this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(fileparts(this_file))));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add shale

[planes, centers] = MulistagePlanarNonPlanarHF( ...
    'numStages', 2, 'fracSpacing', 10, 'numFracsPerStage', 3, ...
    'fracHalfLength', [4, 5], 'fracHeight', [6, 8], 'theta', 0.4, ...
    'clusterSpacing', 2, 'heelCoord', [10, 20, 15]);
multistage_points = cat(3, planes.points);
multistage_centers = centers;

Gs = StructuredGridFractureonFace([100, 200, 60], centers, ...
    'numStages', 2, 'numFracsPerStage', 3, 'fracSpacing', 10, ...
    'nxRefine', 10, 'nxRefine_small', 2, 'ny', 11, 'nz', 11);
structured_x = unique(Gs.nodes.coords(:, 1));
structured_y = unique(Gs.nodes.coords(:, 2));
structured_z = unique(Gs.nodes.coords(:, 3));
structured_dims = Gs.cartDims;

[G1, ids1, wells1] = explicitStencil([100, 300, 60], ...
    'fracHalfLength', 100, 'gridType', 'geomspacing');
explicit_x = unique(G1.nodes.coords(:, 1));
explicit_y = unique(G1.nodes.coords(:, 2));
explicit_z = unique(G1.nodes.coords(:, 3));
explicit_dims = G1.cartDims;
explicit_ids = ids1;
explicit_wells = wells1;

[G3, ids3, wells3] = explicitStencil3Fracs([100, 300, 60], ...
    'fracHalfLength', 100, 'gridType', 'geomspacing');
close all
explicit3_x = unique(G3.nodes.coords(:, 1));
explicit3_y = unique(G3.nodes.coords(:, 2));
explicit3_z = unique(G3.nodes.coords(:, 3));
explicit3_dims = G3.cartDims;
explicit3_ids = ids3;
explicit3_wells = wells3;

save(output_path, 'multistage_points', 'multistage_centers', ...
    'structured_x', 'structured_y', 'structured_z', 'structured_dims', ...
    'explicit_x', 'explicit_y', 'explicit_z', 'explicit_dims', ...
    'explicit_ids', 'explicit_wells', 'explicit3_x', 'explicit3_y', ...
    'explicit3_z', 'explicit3_dims', 'explicit3_ids', 'explicit3_wells', '-v7');
fprintf('Wrote shale SPE geometry fixture to %s\n', output_path);
end
