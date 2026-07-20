%% Dump Phase 12 reference: coarsegrid + flow-based permeability upscaling.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add coarsegrid upscaling incomp mimetic

fine = [12, 12, 6];
G = computeGeometry(cartGrid(fine, [120, 120, 60]));

nc = G.cells.num;
kvals = (1 + mod(13 * (0 : nc - 1)', 200)) * milli * darcy;
rock  = makeRock(G, kvals, 0.1 + 0.2 * mod((0 : nc - 1)', 5) / 4);

coarse = [3, 3, 2];
p  = partitionCartGrid(G.cartDims, coarse);
cg = generateCoarseGrid(G, p);

K = upscalePerm(G, cg, rock, 'method', 'tpfa');

pv    = poreVolume(G, rock);
poroU = accumarray(p, pv) ./ accumarray(p, G.cells.volumes);

partition = p;
save(fullfile(thisDir, 'ref_p12_upscaling.mat'), '-v7', ...
     'fine', 'coarse', 'kvals', 'K', 'poroU', 'partition');
fprintf('Wrote Phase-12 upscaling reference (%d blocks)\n', cg.cells.num);
