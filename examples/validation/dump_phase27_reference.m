%% Dump Phase 27 reference: coupled flow and geomechanics (ad-mechanics).
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p27_poromech.mat into this folder; then run
% validate_phase27.py to compare PyMRST against it.
%
% A 3D poroelastic box: sides and base on rollers, a compressive traction on
% top, drained through the top face. Starting from an unloaded, undeformed,
% zero-pressure state, the first step generates the undrained response and
% the rest consolidate it away - so the dump exercises the coupling in both
% regimes.

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech

gravity off;

thisDir = fileparts(mfilename('fullpath'));

%% Grid and rock
nx = 4; ny = 3; nz = 6;
L  = [4, 3, 10];
G = computeGeometry(cartGrid([nx, ny, nz], L));
G = createAugmentedGrid(G);

perm  = 100 * milli * darcy;
poro  = 0.25;
alpha = 1;
rock.perm  = perm  * ones(G.cells.num, 1);
rock.poro  = poro  * ones(G.cells.num, 1);
rock.alpha = alpha * ones(G.cells.num, 1);

%% Mechanics: E, nu, boundary conditions
E  = 1 * giga * Pascal;
nu = 0.25;
mech_problem.E    = E  * ones(G.cells.num, 1);
mech_problem.nu   = nu * ones(G.cells.num, 1);
mech_problem.load = @(x) 0 * x;

pts = G.nodes.coords;
tol = 1e-9;

% roller conditions: normal displacement fixed on the sides and the base
xmin = find(pts(:,1) < tol);         xmax = find(pts(:,1) > L(1) - tol);
ymin = find(pts(:,2) < tol);         ymax = find(pts(:,2) > L(2) - tol);
base = find(pts(:,3) > L(3) - tol);  % z increases downward in MRST cartGrid

nodes = [xmin; xmax; ymin; ymax; base];
mask  = [repmat([true,  false, false], numel(xmin) + numel(xmax), 1); ...
         repmat([false, true,  false], numel(ymin) + numel(ymax), 1); ...
         repmat([false, false, true ], numel(base), 1)];

disp_bc = struct('nodes', nodes, ...
                 'uu',    zeros(numel(nodes), 3), ...
                 'mask',  mask);

% compressive traction on the top face (z = 0)
fc = G.faces.centroids;
bf = any(G.faces.neighbors == 0, 2);
top_faces = find(bf & fc(:,3) < tol);
sigma0 = 10 * mega * Pascal;
force_bc = struct('faces', top_faces, ...
                  'force', repmat([0, 0, sigma0], numel(top_faces), 1));

mech_problem.el_bc = struct('disp_bc', disp_bc, 'force_bc', force_bc);

%% Fluid: slightly compressible water
muW  = 1 * centi * poise;
rhoW = 1000 * kilogram / meter^3;
cW   = 1e-8 / barsa;
pRef = 0;
fluid = initSimpleADIFluid('phases', 'W', 'mu', muW, 'rho', rhoW, ...
                           'c', cW, 'pRef', pRef);

%% Flow BC: drained through the top face
bc = addBC([], top_faces, 'pressure', 0);

%% Coupled model
model = MechWaterModel(G, rock, fluid, mech_problem);

% MRST's defaults (toleranceCNV = 1e-3, toleranceMB = 1e-7) stop Newton well
% before the discrete system is actually solved, and the leftover residual
% accumulates over the schedule - it is what put a 2e-7 relative gap between
% the two codes until this was tightened. Solve the system properly, so the
% comparison is between two discretizations rather than between two stopping
% rules.
model.toleranceCNV       = 1e-12;
model.toleranceMB        = 1e-14;
model.nonlinearTolerance = 1e-12;

mech_unknowns = ~model.mechModel.operators.isdirdofs;
initState.pressure = zeros(G.cells.num, 1);
initState.xd       = zeros(nnz(mech_unknowns), 1);
initState = addDerivedQuantities(model.mechModel, initState);

%% Schedule: geometric-ish steps through the consolidation
dt = [1, 1, 5, 10, 50, 100, 500] * second;
schedule.step.val     = dt(:);
schedule.step.control = ones(numel(dt), 1);
schedule.control      = struct('W', [], 'bc', bc);

[~, states] = simulateScheduleAD(initState, model, schedule);

%% Collect
nsteps  = numel(states);
p_hist  = zeros(nsteps, G.cells.num);
xd_hist = zeros(nsteps, nnz(mech_unknowns));
uz_hist = zeros(nsteps, G.nodes.num);
for i = 1 : nsteps
    p_hist(i, :)  = states{i}.pressure(:)';
    xd_hist(i, :) = states{i}.xd(:)';
    uu            = reshape(states{i}.u, 3, [])';
    uz_hist(i, :) = uu(:, 3)';
end

WI = [];  %#ok<NASGU>  (no wells in this case)
save(fullfile(thisDir, 'ref_p27_poromech.mat'), '-v7', ...
     'p_hist', 'xd_hist', 'uz_hist', 'dt', 'nx', 'ny', 'nz', 'L', ...
     'E', 'nu', 'alpha', 'perm', 'poro', 'muW', 'rhoW', 'cW', 'sigma0');

fprintf('Wrote ref_p27_poromech.mat (%d steps) to %s\n', nsteps, thisDir);
