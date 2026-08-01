%% Reference data for compactionTest2D: computed field vs its analytic solution

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

opt = struct('E', 0.3e9, 'nu', 0.4, 'islinear', false, ...
             'hanging', false, 'free_top', true, ...
             'gravity_load', true, 'top_load', false);
cartDims = [10 10];
L = [15 15];

results = struct();
for k = 1:2
    triangulate = (k == 2);
    G = squareGrid(cartDims, L, 'grid_type', 'cartgrid');
    if triangulate
        face = unique(G.cells.faces(any(bsxfun(@eq, G.cells.faces(:, 2), [3, 4]), 2), 1));
        G = triangulateFaces(G, face);
        G = sortEdges(G);
    end
    G = createAugmentedGrid(G);
    G = computeGeometry(G);

    Ev  = repmat(opt.E, G.cells.num, 1);
    nuv = repmat(opt.nu, G.cells.num, 1);
    C   = Enu2C(Ev, nuv, G);

    [el_bc, load] = makeCompactionTest(G, opt);
    uu = VEM_linElast(G, C, el_bc, load);

    if (~isempty(el_bc.force_bc))
        ff = abs(el_bc.force_bc.force(1, G.griddim));
    else
        ff = 0;
    end
    start = max(G.faces.centroids(:, G.griddim));
    top   = min(G.faces.centroids(:, G.griddim));
    fac   = 100*300/2;
    ana = @(z) ff*(z-start)./(C(1, 1)) ...
          - double(opt.gravity_load)*fac*((top-start).^2 - (z-top).^2)/C(1, 1);

    z = G.nodes.coords(:, G.griddim);
    name = 'plain';
    if triangulate
        name = 'triangulated';
    end
    results.([name '_cells'])    = G.cells.num;
    results.([name '_nodes'])    = G.nodes.num;
    results.([name '_uz_min'])   = min(uu(:, G.griddim));
    results.([name '_ana_min'])  = min(ana(z));
    results.([name '_C11'])      = C(1, 1);
    results.([name '_uz'])       = uu(:, G.griddim);
    results.([name '_ana'])      = ana(z);
    results.([name '_coords'])   = G.nodes.coords;
end

plain_cells = results.plain_cells;
plain_uz_min = results.plain_uz_min;
plain_ana_min = results.plain_ana_min;
plain_uz = results.plain_uz;
plain_ana = results.plain_ana;
plain_coords = results.plain_coords;
tri_cells = results.triangulated_cells;
tri_nodes = results.triangulated_nodes;
tri_uz_min = results.triangulated_uz_min;
tri_ana_min = results.triangulated_ana_min;
tri_uz = results.triangulated_uz;
tri_ana = results.triangulated_ana;
tri_coords = results.triangulated_coords;
C11 = results.plain_C11;

fprintf('plain       : computed %.6e  analytic %.6e  ratio %.4f\n', ...
        plain_uz_min, plain_ana_min, plain_uz_min/plain_ana_min);
fprintf('triangulated: computed %.6e  analytic %.6e  ratio %.4f\n', ...
        tri_uz_min, tri_ana_min, tri_uz_min/tri_ana_min);

save(fullfile(thisDir, 'ref_vemmech_compaction.mat'), '-v7', ...
     'plain_cells', 'plain_uz_min', 'plain_ana_min', 'plain_uz', 'plain_ana', ...
     'plain_coords', 'tri_cells', 'tri_nodes', 'tri_uz_min', 'tri_ana_min', ...
     'tri_uz', 'tri_ana', 'tri_coords', 'C11');
fprintf('Wrote compaction reference to %s\n', thisDir);
