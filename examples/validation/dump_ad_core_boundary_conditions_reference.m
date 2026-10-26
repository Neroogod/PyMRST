% The boundary-condition and source utilities in MRST 2024a, on AD inputs,
% for validate_ad_core_boundary_conditions.py: getBoundaryConditionFluxesAD
% (pressure, flux and reservoir-rate faces, zero saturations, outside rho and
% mob), getSourceFluxesAD, computeSourcesAndBoundaryConditionsAD (with and
% without dissolution) and addFluxesFromSourcesAndBC.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props sequential
gravity reset on
G = computeGeometry(cartGrid([4, 3, 2], [40, 30, 20]));
nc = G.cells.num;
rock = makeRock(G, (1 + (1:nc)')*1e-13, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 5]*centi*poise, 'rho', [1000, 800], 'n', [2, 2]);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
model.AutoDiffBackend = AutoDiffBackend();
[p, sw] = initVariablesADI(1e7 + 1e5*(1:nc)', 0.2 + 0.6*(1:nc)'/nc);
so = 1 - sw;
mob = {sw.^2/1e-3, so.^2/5e-3};
rho = {1000*exp(1e-9*(p - 1e7)), 800*exp(5e-9*(p - 1e7))};
b = {rho{1}/1000, rho{2}/800};
pressure = {p, p}; s = {sw, so};
bnd = find(any(G.faces.neighbors == 0, 2));
faces = bnd([1, 2, 5, 9, 14, 20, 27, 33]);
bc = addBC([], faces(1:2), 'pressure', [1.2e7; 0.9e7], 'sat', [1, 0]);
bc = addBC(bc, faces(3:4), 'flux', [2e-4; -3e-4], 'sat', [0, 1]);
bc = addBC(bc, faces(5:6), 'pressure', [1.05e7; 1.4e7], 'sat', [0, 0]);
bc = addBC(bc, faces(7:8), 'flux', [1e-4; 5e-5], 'sat', [0.3, 0.7]);
bc.type(7:8) = {'rate'};
pack = @(q) struct('val', value(q), 'jp', full(q.jac{1}), 'js', full(q.jac{2}));
packAll = @(c) cellfun(pack, c, 'UniformOutput', false);
[qS, map, cells, qR] = getBoundaryConditionFluxesAD(model, pressure, s, mob, rho, b, bc);
bc1 = struct('qS', {packAll(qS)}, 'qR', {packAll(qR)}, 'cells', cells, 'map', full(map));
bco = bc; bco.rho = [1010*ones(8, 1), 790*ones(8, 1)]; bco.mob = [100*ones(8, 1), 30*ones(8, 1)];
bco.type(7:8) = {'flux'};
[qS, ~, ~, qR] = getBoundaryConditionFluxesAD(model, pressure, s, mob, rho, b, bco);
bc2 = struct('qS', {packAll(qS)}, 'qR', {packAll(qR)});
src = addSource([], [1; 5; 9], [1e-4; -2e-4; -1e-4], 'sat', [1, 0; 0, 0; 0, 0]);
[qS, smap, scells] = getSourceFluxesAD(model, mob, s, src);
srcs = struct('qS', {packAll(qS)}, 'cells', scells, 'map', full(smap));
srcInj = addSource([], [2; 3], [1e-4; 2e-4], 'sat', [1, 0; 0.5, 0.5]);
qS = getSourceFluxesAD(model, mob, s, srcInj);
srcInjOnly = struct('val', {cellfun(@double, qS, 'UniformOutput', false)}, ...
                    'isAD', cellfun(@(x) isa(x, 'ADI'), qS));
forces = struct('bc', bc, 'src', src);
[cs, cb] = computeSourcesAndBoundaryConditionsAD(model, pressure, s, mob, rho, {}, forces);
packC = @(c) struct('mass', {packAll(c.phaseMass)}, 'vol', {packAll(c.phaseVolume)}, ...
                    'mapping', full(c.mapping), 'cells', c.sourceCells);
contrib = struct('src', packC(cs), 'bc', packC(cb));
eqs = {sw*1e-3, so*2e-3};
[eqsOut, ~, ~, ~, ~, srcCells, bcCells] = addFluxesFromSourcesAndBC(model, eqs, pressure, rho, mob, s, forces);
added = struct('eqs', {packAll(eqsOut)}, 'srcCells', srcCells, 'bcCells', bcCells);
% Three phases with gas dissolved in oil
fluid3 = initSimpleADIFluid('phases', 'WOG', 'mu', [1, 5, 0.2]*centi*poise, 'rho', [1000, 800, 100], 'n', [2, 2, 2]);
model3 = GenericBlackOilModel(G, rock, fluid3, 'disgas', true);
model3.AutoDiffBackend = AutoDiffBackend();
sg = 0.1 + 0*sw; so3 = 1 - sw - sg;
mob3 = {sw.^2/1e-3, so3.^2/5e-3, sg.^2/2e-4};
rho3 = {rho{1}, rho{2}, 100*exp(1e-8*(p - 1e7))};
rs = 50 + 1e-6*(p - 1e7);
dissolved = {{[], [], []}, {[], [], []}, {[], rs, []}};
bc3 = addBC([], faces(1:3), 'pressure', [1.2e7; 0.9e7; 1.3e7], 'sat', [0, 0.5, 0.5]);
src3 = addSource([], [1; 5], [1e-4; -2e-4], 'sat', [0, 0.5, 0.5; 0, 0, 0]);
src3.dissolution = reshape([0 0 0 0 0 0 0 30 0], 3, 3);
forces3 = struct('bc', bc3, 'src', src3);
[cs3, cb3] = computeSourcesAndBoundaryConditionsAD(model3, {p, p, p}, {sw, so3, sg}, mob3, rho3, dissolved, forces3);
contrib3 = struct('src', packC(cs3), 'bc', packC(cb3));
save(fullfile(here, 'ref_ad_core_boundary_conditions.mat'), '-v7', 'faces', 'bc1', 'bc2', 'srcs', ...
     'srcInjOnly', 'contrib', 'added', 'contrib3');
fprintf('saved: %d bc faces\n', numel(faces));
