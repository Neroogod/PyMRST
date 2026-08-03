function dump_shale_foundation_reference(output_path)
% Direct MRST fixtures for deterministic PyMRST shale foundation owners.

if nargin ~= 1
    error('Expected explicit output_path.');
end
this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(fileparts(this_file))));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add shale

G = computeGeometry(cartGrid([2, 2, 1], [3, 5, 4]));
points = [0.25, 0.25, 1; 2, 0.25, 1; 0.25, 3, 1; 2, 3, 1];
point_cell_ids_one_based = Pts2CellIdx(points, G);

[fracplanes, frac_centroids] = createMultiStageHFs( ...
    'numStages', 2, 'fracSpacing', 10, 'numFracsPerStage', 2, ...
    'fracHalfLength', [4, 5], 'fracHeight', [6, 8], ...
    'clusterSpacing', 2, 'fracCentroid1', [10, 20, 15]);
fracture_points = cat(3, fracplanes.points);

pressure = [0; 2; 4];
gangi = gangiFn(pressure, 10, 0.5, 20, 2, 4, 2);

rock = makeShaleRock(G, [10, 5, 1], [0.2; 0.1; 0.3; 0.4], 'ntg', 0.8);
rock_perm = rock.perm;
rock_poro = rock.poro;
rock_ntg = rock.ntg;
rock_is_matrix = rock.isMatrix;

save(output_path, 'point_cell_ids_one_based', 'frac_centroids', ...
    'fracture_points', 'gangi', 'rock_perm', 'rock_poro', 'rock_ntg', ...
    'rock_is_matrix', '-v7');
fprintf('Wrote shale foundation fixture to %s\n', output_path);
end
