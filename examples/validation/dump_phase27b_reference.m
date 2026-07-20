%% Dump Phase 27b reference: multiphase poroelasticity (MechOilWaterModel).
%
% Run in MATLAB after starting MRST. Writes ref_p27b_mechow.mat.
%
% The SAME case is run on two grids that differ only in cell size:
%
%   'big'  - cells of 10 x 10 x 5 m   (volume 500 m^3)
%   'unit' - cells of  1 x  1 x 1 m   (volume 1 m^3), everything else scaled
%            so the physics is the same problem
%
% Why: `equationsOilWaterMech.m` computes
%
%     effPorVol = rock.poro .* (G.cells.volumes .* pvMult) + rock.alpha .* mechTerm
%
% where mechTerm = div(u) is dimensionless - so the cell volume is *missing*
% from the Biot term, unlike in `equationsWaterMech.m` and
% `equationsBlackOilMech.m`, which both write
%
%     effPorVol = G.cells.volumes .* (poro .* pvMult + alpha .* mechTerm).
%
% If that is really the defect, then MRST must agree with a correct
% implementation on the unit grid (where the missing factor is 1) and
% disagree on the big one. That is what validate_phase27.py checks.

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech

gravity off;
thisDir = fileparts(mfilename('fullpath'));

nx = 5; ny = 5; nz = 3;
E = 1 * giga * Pascal;  nu = 0.25;  alpha = 1;
perm = 100 * milli * darcy;  poro = 0.25;
p0 = 200 * barsa;  sw0 = 0.2;
qinj = 1e-3;  pbhp = 150 * barsa;
dt = [30, 30, 30] * day;

out = struct();
cases = {'big', 'unit'};
sizes = {[10, 10, 5], [1, 1, 1]};

for ci = 1 : numel(cases)
    cs = sizes{ci};
    G = computeGeometry(cartGrid([nx, ny, nz], [nx, ny, nz] .* cs));
    G = createAugmentedGrid(G);
    nc = G.cells.num;

    rock.perm  = perm  * ones(nc, 1);
    rock.poro  = poro  * ones(nc, 1);
    rock.alpha = alpha * ones(nc, 1);

    mech_problem.E    = E  * ones(nc, 1);
    mech_problem.nu   = nu * ones(nc, 1);
    mech_problem.load = @(x) 0 * x;

    pts = G.nodes.coords;  tol = 1e-9;  L = [nx, ny, nz] .* cs;
    xf = find(pts(:,1) < tol | pts(:,1) > L(1) - tol);
    yf = find(pts(:,2) < tol | pts(:,2) > L(2) - tol);
    zf = find(pts(:,3) > L(3) - tol);
    nodes = [xf; yf; zf];
    mask  = [repmat([true, false, false], numel(xf), 1); ...
             repmat([false, true, false], numel(yf), 1); ...
             repmat([false, false, true], numel(zf), 1)];
    mech_problem.el_bc = struct( ...
        'disp_bc', struct('nodes', nodes, 'uu', zeros(numel(nodes), 3), ...
                          'mask', mask), ...
        'force_bc', []);

    fluid = initSimpleADIFluid('phases', 'WO', ...
                               'mu',   [1, 5] * centi * poise, ...
                               'rho',  [1000, 800], ...
                               'n',    [2, 2], ...
                               'c',    [1e-6, 1e-4] / barsa, ...
                               'pRef', p0);

    % This MRST tree carries a local modification ("OMO") in
    % SimpleWell.updateConnectionPressureDrop that reads fluid.rhoSinj for
    % any well with sign == 1. initSimpleADIFluid does not set it, so an
    % injector cannot run without this line.
    fluid.rhoSinj = [1000, 800];

    W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', qinj, ...
                'Comp_i', [1, 0], 'Sign', 1, 'Radius', 0.1);
    W = addWell(W, G, rock, nc, 'Type', 'bhp', 'Val', pbhp, ...
                'Comp_i', [0, 1], 'Sign', -1, 'Radius', 0.1);

    model = MechOilWaterModel(G, rock, fluid, mech_problem);
    model.toleranceCNV       = 1e-12;
    model.toleranceMB        = 1e-14;
    model.nonlinearTolerance = 1e-12;

    mech_unknowns = ~model.mechModel.operators.isdirdofs;
    initState.pressure = p0  * ones(nc, 1);
    initState.s        = repmat([sw0, 1 - sw0], nc, 1);
    initState.xd       = zeros(nnz(mech_unknowns), 1);
    initState = addDerivedQuantities(model.mechModel, initState);
    % wellSol is left for validateState to build - constructing it here gives
    % an empty cqs and SimpleWell then fails on the first iteration.

    schedule.step.val     = dt(:);
    schedule.step.control = ones(numel(dt), 1);
    schedule.control      = struct('W', W);

    [~, states] = simulateScheduleAD(initState, model, schedule);

    ns = numel(states);
    p_hist  = zeros(ns, nc);
    sw_hist = zeros(ns, nc);
    xd_hist = zeros(ns, nnz(mech_unknowns));
    bhp     = zeros(ns, numel(W));
    for i = 1 : ns
        p_hist(i, :)  = states{i}.pressure(:)';
        sw_hist(i, :) = states{i}.s(:, 1)';
        xd_hist(i, :) = states{i}.xd(:)';
        bhp(i, :)     = [states{i}.wellSol.bhp];
    end

    out.(cases{ci}) = struct('p', p_hist, 'sw', sw_hist, 'xd', xd_hist, ...
                             'bhp', bhp, 'cellsize', cs, ...
                             'cellvol', G.cells.volumes(1));
    clear initState schedule rock mech_problem;
end

big = out.big;  unit = out.unit;  %#ok<NASGU>
save(fullfile(thisDir, 'ref_p27b_mechow.mat'), '-v7', 'big', 'unit', ...
     'nx', 'ny', 'nz', 'E', 'nu', 'alpha', 'perm', 'poro', 'p0', 'sw0', ...
     'qinj', 'pbhp', 'dt');

fprintf('Wrote ref_p27b_mechow.mat to %s\n', thisDir);
fprintf('  big  cell volume: %g m^3\n', out.big.cellvol);
fprintf('  unit cell volume: %g m^3\n', out.unit.cellvol);
