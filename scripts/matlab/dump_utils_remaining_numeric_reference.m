% MRST references for the remaining deterministic utility numerics.
repo = '/Users/Shared/PyMRST_workspace';
mrst = fullfile(repo, 'mrst-2024a-Geomech');
addpath(fullfile(mrst, 'utils'));
addpath(fullfile(mrst, 'utils', 'ad_utils'));
addpath(fullfile(mrst, 'utils', 'units'));
addpath(fullfile(mrst, 'gridprocessing'));
addpath(fullfile(mrst, 'params'));

diag_values = [2; 0; -1; 4];
jac_a = sparse([1 0 2; 3 4 0; 5 0 6; 0 7 8]);
jac_b = sparse([1 2; 3 4; 5 6; 7 8]);
scaled = lMultdiag(diag_values, {jac_a, jac_b});
scaled_a = full(scaled{1}); scaled_b = full(scaled{2});

G2 = computeGeometry(cartGrid([3, 2], [6, 10]));
dims_cells = [1; 4; 6];
[dims_x, dims_y, dims_z] = cellDimsCG(G2, dims_cells);
sorted_cell_nodes = getSortedCellNodes(G2);

gravity reset on
G3 = computeGeometry(cartGrid([1, 1, 8], [10, 10, 80]));
contacts = [30; 55]; densities = [700; 850; 1000];
[hydro_p, hydro_fn] = incompHydrostaticPressure(...
    G3, contacts, densities, 'topPressure', 1.25e5, 'nsamples', 41);
hydro_query_z = [5; 20; 30; 40; 55; 70; 90];
hydro_query_p = hydro_fn(hydro_query_z);
hydro_depth = G3.cells.centroids(:, 3);

coarse_plain = isCoarseGrid(struct());
coarse_both = isCoarseGrid(struct('parent', [], 'partition', []));
coarse_parent = isCoarseGrid(struct('parent', []));

unit_names = {'metric', 'field', 'lab', 'si'};
unit_values = zeros(4, 7);
for k = 1:numel(unit_names)
    u = getUnitSystem(unit_names{k});
    unit_values(k, :) = [u.length, u.time, u.press, u.viscosity, ...
                         u.perm, u.resvolume, u.trans];
end

destination = fullfile(repo, 'PyMRST', 'scripts', 'matlab', ...
                       'ref_utils_remaining_numeric.mat');
save(destination, 'diag_values', 'jac_a', 'jac_b', 'scaled_a', 'scaled_b', ...
     'dims_cells', 'dims_x', 'dims_y', 'dims_z', 'sorted_cell_nodes', ...
     'contacts', 'densities', 'hydro_p', 'hydro_query_z', ...
     'hydro_query_p', 'hydro_depth', 'coarse_plain', 'coarse_both', ...
     'coarse_parent', 'unit_names', 'unit_values', '-v7');
fprintf('wrote %s\n', destination);
