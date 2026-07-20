%% Dump the Norne coupled geomechanics case from MRST, for validate_norne.py.
%
% Mirrors examples/run_norne_geomech.py cell for cell: the same cut of the
% same field, same rock, same E/nu/alpha, same boundary conditions, same
% fluid, same wells, same schedule. Run in MATLAB after starting MRST.
%
% Uses MRST's MechOilWaterModel - which is the model whose Biot term is
% missing G.cells.volumes (see README.md, seventh finding). So PyMRST's
% *correct* model is expected to disagree with this, and PyMRST run with
% `mrst_oilwater_bug=True` is expected to reproduce it. validate_norne.py
% checks both.

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech deckformat

gravity on;
thisDir = fileparts(mfilename('fullpath'));

% Use the same Norne files PyMRST downloaded, so neither side re-downloads.
pyData = fullfile(getenv('HOME'), '.pymrst', 'datasets');
if exist(fullfile(pyData, 'norne', 'INCLUDE', 'IRAP_1005.GRDECL'), 'file')
    mrstDataDirectory(pyData);
end
assert(makeNorneSubsetAvailable() && makeNorneGRDECL(), ...
       'Could not obtain the Norne dataset');

grdecl = readGRDECL(fullfile(getDatasetPath('norne'), 'NORNE.GRDECL'));
grdecl = convertInputUnits(grdecl, getUnitSystem('METRIC'));

% mini Norne, exactly as runNorneExample.m cuts it
grdecl = cutGrdecl(grdecl, [10 20; 35 45; 1 5]);
grdecl.ACTNUM = ones(size(grdecl.ACTNUM));

G = processGRDECL(grdecl);
G = computeGeometry(G(1));
G = createAugmentedGrid(G);
nc = G.cells.num;

perm = [grdecl.PERMX, grdecl.PERMY, grdecl.PERMZ];
rock.perm  = perm(G.cells.indexMap, :);
rock.poro  = max(grdecl.PORO(G.cells.indexMap), 0.1);
rock.alpha = ones(nc, 1);

pRef = 270 * barsa;
E    = 1 * giga * Pascal;
nu   = 0.3;
mech_problem.E    = E  * ones(nc, 1);
mech_problem.nu   = nu * ones(nc, 1);
mech_problem.load = @(x) 0 * x;

%% Boundary conditions: base clamped, everything else confined at pRef.
% Selected by the face normal, not by depth - Norne's base is not flat.
nb   = G.faces.neighbors;
isbd = nb(:,1) == 0 | nb(:,2) == 0;
sgn  = double(nb(:,1) == 0) - double(nb(:,2) == 0);
n_in = bsxfun(@times, G.faces.normals, sgn ./ G.faces.areas);

bottomfaces = find(isbd & (n_in(:,3) < -0.5));
bn = G.faces.nodes(mcolon(G.faces.nodePos(bottomfaces), ...
                          G.faces.nodePos(bottomfaces + 1) - 1));
bcnodes = unique(bn);
disp_bc = struct('nodes', bcnodes, ...
                 'uu',    zeros(numel(bcnodes), 3), ...
                 'mask',  true(numel(bcnodes), 3));

outer = find(isbd);
outer = outer(~ismember(outer, bottomfaces));
force_bc = struct('faces', outer, 'force', pRef * n_in(outer, :));

mech_problem.el_bc = struct('disp_bc', disp_bc, 'force_bc', force_bc);

%% Fluid, wells, schedule - runNorneExample.m's 'oil water' case
fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu',   [1, 10] * centi * poise, ...
                           'n',    [1, 1], ...
                           'rho',  [1000, 700] * kilogram / meter^3, ...
                           'c',    [1e-10, 1e-10], ...
                           'pRef', pRef);
fluid.rhoSinj = [1000, 700];   % this tree's local SimpleWell modification

refdepth = G.cells.centroids(1, 3);
W = addWell([], G, rock, 10, 'Type', 'rate', 'Val', 1e4/day, ...
            'Sign', 1, 'Comp_i', [1, 0], 'Name', 'inj', 'refDepth', refdepth);
W = addWell(W, G, rock, nc, 'Type', 'bhp', 'Val', pRef, ...
            'Sign', -1, 'Comp_i', [0, 1], 'Name', 'prod', 'refDepth', refdepth);

model = MechOilWaterModel(G, rock, fluid, mech_problem);
model.toleranceCNV       = 1e-10;
model.toleranceMB        = 1e-13;
model.nonlinearTolerance = 1e-10;

dt = [1*day; 5*day*ones(20, 1)];
schedule.step.val     = dt;
schedule.step.control = ones(numel(dt), 1);
schedule.control      = struct('W', W);

initState.pressure = pRef * ones(nc, 1);
initState.s        = repmat([0, 1], nc, 1);          % starts full of oil
initState = computeInitDisp(model, initState, [], 'pressure', initState.pressure);

[~, states] = simulateScheduleAD(initState, model, schedule);

%% Collect
ns = numel(states);
p_hist  = zeros(ns, nc);
sw_hist = zeros(ns, nc);
u_hist  = zeros(ns, 3 * G.nodes.num);
uz_hist = zeros(ns, G.nodes.num);
divu_hist = zeros(ns, nc);
strain_hist = zeros(ns, nc, 6);
eff_stress_hist = zeros(ns, nc, 6);
total_stress_hist = zeros(ns, nc, 6);
bhp     = zeros(ns, numel(W));
trace_voigt = [1, 1, 1, 0, 0, 0];
for i = 1 : ns
    p_hist(i, :)  = states{i}.pressure(:)';
    sw_hist(i, :) = states{i}.s(:, 1)';
    uvec          = states{i}.u(:);
    u_hist(i, :)  = uvec';
    uu            = reshape(uvec, 3, [])';
    uz_hist(i, :) = uu(:, 3)';
    divu_hist(i, :) = (model.mechModel.operators.ovol_div * uvec ...
                       ./ G.cells.volumes)';
    strain = reshape(model.mechModel.operators.global_strain * uvec, 6, [])';
    eff_stress = reshape(model.mechModel.operators.global_stress * uvec, 6, [])';
    pressure_stress = bsxfun(@times, rock.alpha .* states{i}.pressure, ...
                             trace_voigt);
    strain_hist(i, :, :) = strain;
    eff_stress_hist(i, :, :) = eff_stress;
    total_stress_hist(i, :, :) = eff_stress - pressure_stress;
    bhp(i, :)     = [states{i}.wellSol.bhp];
end
u0 = reshape(initState.u, 3, [])';
uz0 = u0(:, 3)';
u0_full = initState.u(:)';
divu0 = (model.mechModel.operators.ovol_div * initState.u(:) ...
         ./ G.cells.volumes)';
strain0 = reshape(model.mechModel.operators.global_strain * initState.u(:), ...
                  6, [])';
eff_stress0 = reshape(model.mechModel.operators.global_stress * initState.u(:), ...
                      6, [])';
total_stress0 = eff_stress0 - bsxfun(@times, rock.alpha .* initState.pressure, ...
                                     trace_voigt);

cellvols = G.cells.volumes;
save(fullfile(thisDir, 'ref_norne_mech.mat'), '-v7', ...
     'p_hist', 'sw_hist', 'u_hist', 'uz_hist', 'uz0', 'u0_full', ...
     'divu_hist', 'divu0', 'strain_hist', 'strain0', ...
     'eff_stress_hist', 'eff_stress0', ...
     'total_stress_hist', 'total_stress0', ...
     'bhp', 'dt', 'cellvols', 'pRef', 'E', 'nu');

fprintf('Wrote ref_norne_mech.mat (%d steps, %d cells) to %s\n', ...
        ns, nc, thisDir);
fprintf('  initial |u| = %g m (should be ~0: pRef confines pRef)\n', ...
        max(abs(initState.u)));
fprintf('  final mean pressure = %g bar\n', mean(p_hist(end,:))/barsa);
fprintf('  max surface uplift  = %g mm\n', ...
        1000 * max(-(uz_hist(end,:) - uz0)));
