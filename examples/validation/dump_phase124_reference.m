%% Dump Phase 124 reference: water transport with frozen well streams.
%
% This isolates MRST's water-active TransportOverallCompositionModel well
% source path. A frozen pressure/flux/wellSol state is prescribed directly,
% matching what the pressure half normally hands to transport.

mrstModule add ad-core ad-props compositional sequential

thisDir = fileparts(mfilename('fullpath'));
gravity reset off;

nx = 8;
G = computeGeometry(cartGrid([nx, 1, 1], [80, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], ...
                           'rho', [1000, 700, 100], ...
                           'mu', [1, 1, 1]*centi*poise, ...
                           'c', [0, 0, 0]);
fluid.rhoSinj = [1000, 700, 100];

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', 5e-5, ...
            'Comp_i', [0.50, 0.25, 0.25], 'Sign', 1, ...
            'components', [0.50, 0.30, 0.20], 'Name', 'I1');
W = addWell(W, G, rock, nx, 'Type', 'bhp', 'Val', 55*barsa, ...
            'Comp_i', [0.20, 0.40, 0.40], 'Sign', -1, ...
            'components', [0.50, 0.30, 0.20], 'Name', 'P1');

modelT = TransportOverallCompositionModel(G, rock, fluid, mixture, ...
                                          'water', true);
modelT.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
modelT.nonlinearTolerance = 1e-10;
try, modelT.toleranceCNV = 1e-10; end %#ok
try, modelT.toleranceMB  = 1e-12; end %#ok

T0 = 350;
p0 = linspace(78*barsa, 62*barsa, nx)';
sW0 = linspace(0.18, 0.34, nx)';
z0 = [0.5, 0.3, 0.2];
s0 = [sW0, 0.5*(1 - sW0), 0.5*(1 - sW0)];

state0 = initCompositionalState(modelT, p0, T0, s0, z0, modelT.EOSModel);
dt = 12*day;
forces = struct('W', W, 'bc', [], 'src', []);
modelT = modelT.validateModel(forces);
state0.wellSol = initWellSolAD(W, modelT, state0);

stateP = state0;
stateP.sT = ones(nx, 1);

% Frozen face phase fluxes in reservoir volume units, one row per face.
flux_pressure = zeros(G.faces.num, 3);
internal = modelT.operators.internalConn;
vt = linspace(5.0e-5, 3.0e-5, nnz(internal))';
flux_pressure(internal, :) = [0.45*vt, 0.25*vt, 0.30*vt];
stateP.flux = flux_pressure;

% Frozen per-perforation well streams. `flux` is reservoir phase volume
% flux; `components` is component mass flux, exactly as transport reads them.
well_flux_pressure = [ 2.50e-5,  1.25e-5,  1.25e-5; ...
                      -1.00e-5, -1.50e-5, -2.50e-5];
well_components_pressure = [ 1.20e-2,  7.00e-3,  3.00e-3; ...
                            -1.10e-2, -8.00e-3, -3.50e-3];
stateP.wellSol(1).flux = well_flux_pressure(1, :);
stateP.wellSol(2).flux = well_flux_pressure(2, :);
stateP.wellSol(1).components = well_components_pressure(1, :);
stateP.wellSol(2).components = well_components_pressure(2, :);
stateP.wellSol(1).qWs = 2.50e-5;
stateP.wellSol(1).qOs = 1.25e-5;
stateP.wellSol(1).qGs = 1.25e-5;
stateP.wellSol(2).qWs = -1.00e-5;
stateP.wellSol(2).qOs = -1.50e-5;
stateP.wellSol(2).qGs = -2.50e-5;
stateP.wellSol(1).bhp = mean(p0);
stateP.wellSol(2).bhp = 55*barsa;

solver = BackslashSolverAD();
maxIts = 40;
tol = 1e-10;
stateT = stateP;
for itsT = 1:maxIts
    [problemT, stateT] = modelT.getEquations(state0, stateT, dt, forces, ...
                                             'iteration', itsT);
    resT = vertcat(value(problemT.equations{:}));
    if max(abs(resT)) < tol
        break
    end
    problemT = problemT.assembleSystem();
    [dxT, ~, ~] = solver.solveLinearProblem(problemT, modelT);
    for varNo = 1:numel(problemT.primaryVariables)
        name = problemT.primaryVariables{varNo};
        dx = dxT{varNo};
        if strcmp(name, 'sT')
            stateT.sT = stateT.sT + dx;
        elseif strcmp(name, 'sW')
            stateT.s(:, 1) = min(max(stateT.s(:, 1) + dx, 0), 1);
        else
            compNo = find(strcmp(names, name));
            if ~isempty(compNo)
                z = stateT.components;
                z0_comp = z(:, compNo);
                if isfinite(modelT.dzMaxAbs)
                    dx = sign(dx).*min(abs(dx), modelT.dzMaxAbs);
                end
                z(:, compNo) = min(max(z0_comp + dx, 0), 1);
                z(:, end) = 1.0 - sum(z(:, 1:end-1), 2);
                z = bsxfun(@rdivide, z, sum(z, 2));
                stateT.components = z;
            end
        end
    end
    stateT = modelT.computeFlash(stateT, dt, 1);
end
stateT_before_update = stateT;
stateT.s = bsxfun(@times, stateT.s, stateT.sT);

p_pressure = stateP.pressure;
z_pressure = stateP.components;
s_pressure = stateP.s;
well_qWs_pressure = vertcat(stateP.wellSol.qWs);
well_qOs_pressure = vertcat(stateP.wellSol.qOs);
well_qGs_pressure = vertcat(stateP.wellSol.qGs);
well_bhp_pressure = vertcat(stateP.wellSol.bhp);

sT_transport = stateT_before_update.sT;
z_transport = stateT.components;
s_transport = stateT.s;
p_transport = stateT.pressure;
well_components_transport = vertcat(stateT_before_update.wellSol.components);
well_qWs_transport = vertcat(stateT_before_update.wellSol.qWs);
well_qOs_transport = vertcat(stateT_before_update.wellSol.qOs);
well_qGs_transport = vertcat(stateT_before_update.wellSol.qGs);
well_bhp_transport = vertcat(stateT_before_update.wellSol.bhp);

inj_WI = W(1).WI;
prod_WI = W(2).WI;

save(fullfile(thisDir, 'ref_p124_comptransport_water_wells.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'sW0', 'z0', 'dt', ...
     'p_pressure', 'z_pressure', 's_pressure', 'flux_pressure', ...
     'well_flux_pressure', 'well_components_pressure', ...
     'well_qWs_pressure', 'well_qOs_pressure', 'well_qGs_pressure', ...
     'well_bhp_pressure', 'sT_transport', 'z_transport', 's_transport', ...
     'p_transport', 'well_components_transport', 'well_qWs_transport', ...
     'well_qOs_transport', 'well_qGs_transport', 'well_bhp_transport', ...
     'inj_WI', 'prod_WI', 'itsT', 'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p124_comptransport_water_wells.mat to %s\n', thisDir);
fprintf('  %d cells, transport %d iterations\n', nx, itsT);
