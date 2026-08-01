%% Exact MRST plotWell mesh fixtures (vertices, quads, and cap polygons).
thisDir = fileparts(mfilename('fullpath'));
workspace = fileparts(fileparts(fileparts(thisDir)));
mrstRoot = fullfile(workspace, 'mrst-2024a-Geomech');
addpath(mrstRoot);
run(fullfile(mrstRoot, 'startup.m'));
set(0, 'DefaultFigureVisible', 'off');

refs = struct();
refs.x = captureWell([2, 1, 1], [2, 1, 1], [1; 2], 'xx', 8);
refs.y = captureWell([1, 2, 1], [1, 2, 1], [1; 2], 'yy', 8);
refs.z = captureWell([1, 1, 2], [1, 1, 2], [1; 2], 'zz', 8);
refs.slanted = captureWell([2, 1, 2], [2, 1, 2], [1; 4], 'qq', 8);
refs.single = captureWell([1, 1, 1], [2, 3, 4], 1, 'z', 10);

save(fullfile(thisDir, 'plot_well_mesh_reference.mat'), '-struct', 'refs');
close all force;

function ref = captureWell(dims, extent, cells, direction, cylpts)
    G = computeGeometry(cartGrid(dims, extent));
    W = struct('cells', cells, 'dir', direction, 'sign', 1, 'name', 'W');
    figure('Visible', 'off');
    [heads, ~, path] = plotWell(G, W, 'cylpts', cylpts, 'fontsize', 0);
    ref.vertices = get(path, 'Vertices');
    ref.faces = get(path, 'Faces');
    ref.head_vertices = get(heads(1), 'Vertices');
    ref.heel_vertices = get(heads(2), 'Vertices');
    close(gcf);
end
