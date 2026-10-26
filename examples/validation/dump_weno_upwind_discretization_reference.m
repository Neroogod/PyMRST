% Reference for WENOUpwindDiscretization and getBoundaryFacesMerged
% (validate_ad_core_weno_upwind_discretization.py): face values and their
% Jacobians in 1D (20 cells), 2D (7x5 cells of 2 m x 0.5 m, so each cell's
% SVD frame is anisotropic) and 3D (4x3x2), for a smooth field and a step,
% with a fixed pseudo-random upwind direction per face.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
mrstSettings('set', 'useMEX', false);
grids = {cartGrid(20, 20), cartGrid([7, 5], [14, 2.5]), cartGrid([4, 3, 2], [4, 3, 2])};
out = struct();
for k = 1:numel(grids)
    G = computeGeometry(grids{k});
    rock = makeRock(G, 1, 1);
    fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], 'rho', [1000, 800], 'mu', [1, 5]);
    model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
    weno = WENOUpwindDiscretization(model);
    x = G.cells.centroids;
    smooth = 0.5 + 0.3*sin(x(:, 1)/3).*cos(sum(x(:, 2:end), 2)/2);
    step = double(x(:, 1) > mean(x(:, 1)));
    nf = size(model.operators.N, 1);
    flag = mod((1:nf)'*7, 5) < 2;
    [bfaces, bcells, bfc] = getBoundaryFacesMerged(G);
    fields = {smooth, step};
    for f = 1:2
        q = initVariablesADI(fields{f});
        v = weno.faceUpstream(model, struct(), flag, q);
        out.(sprintf('val_g%d_f%d', k, f)) = value(v);
        out.(sprintf('jac_g%d_f%d', k, f)) = full(v.jac{1});
    end
    out.(sprintf('flag_g%d', k)) = flag;
    out.(sprintf('bfaces_g%d', k)) = bfaces;
    out.(sprintf('bcells_g%d', k)) = bcells;
    out.(sprintf('bfc_g%d', k)) = bfc;
end
save(fullfile(here, 'ref_weno_upwind_discretization.mat'), '-struct', 'out', '-v7');
