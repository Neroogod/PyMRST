%% Dump Phase 123 reference: sequential transport with frozen well streams.
%
% This validates MRST's no-water TransportOverallCompositionModel with wells.
% The transport half consumes wellSol.flux and wellSol.components from the
% pressure solve and does not solve qOs/qGs/bhp as transport unknowns.

mrstModule add ad-core ad-props compositional sequential

thisDir = fileparts(mfilename('fullpath'));
gravity reset off;

% -- grid, rock ---------------------------------------------------------------
nx = 8;
G = computeGeometry(cartGrid([nx, 1, 1], [80, 10, 10]));
rock = makeRock(G, 100*milli*darcy, 0.25);

% -- fluid, mixture, wells ----------------------------------------------------
fluid = initSimpleADIFluid('phases', 'OG', 'n', [2, 2], ...
                           'rho', [700, 100], ...
                           'mu', [1, 1]*centi*poise, ...
                           'c', [0, 0]);
fluid.rhoSinj = [700, 100];

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];
mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);

W = addWell([], G, rock, 1, 'Type', 'rate', 'Val', 5e-5, ...
            'Comp_i', [0, 1], 'Sign', 1, ...
            'components', [0.90, 0.05, 0.05], 'Name', 'I1');
W = addWell(W, G, rock, nx, 'Type', 'bhp', 'Val', 55*barsa, ...
            'Comp_i', [0, 1], 'Sign', -1, ...
            'components', [0.50, 0.30, 0.20], 'Name', 'P1');

modelP = PressureOverallCompositionModel(G, rock, fluid, mixture, ...
                                         'water', false);
modelP.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
modelP.nonlinearTolerance = 1e-10;
try, modelP.toleranceCNV = 1e-10; end %#ok
try, modelP.toleranceMB  = 1e-12; end %#ok

modelT = TransportOverallCompositionModel(G, rock, fluid, mixture, ...
                                          'water', false);
modelT.EOSModel = EquationOfStateModel([], mixture, 'Peng-Robinson');
modelT.nonlinearTolerance = 1e-10;
try, modelT.toleranceCNV = 1e-10; end %#ok
try, modelT.toleranceMB  = 1e-12; end %#ok

% -- previous state and pressure solve ---------------------------------------
T0 = 350;
p0 = linspace(78*barsa, 62*barsa, nx)';
p_guess = p0 + linspace(0.6*barsa, -0.4*barsa, nx)';
z0 = [0.5, 0.3, 0.2];

state0 = initCompositionalState(modelP, p0, T0, [0.5, 0.5], z0, modelP.EOSModel);

dt = 12*day;
forces = struct('W', W, 'bc', [], 'src', []);
modelP = modelP.validateModel(forces);
modelT = modelT.validateModel(forces);
state0.wellSol = initWellSolAD(W, modelP, state0);
stateP = state0;
stateP.pressure = p_guess;
solver = BackslashSolverAD();

maxIts = 40;
tol = 1e-10;
for itsP = 1:maxIts
    [problemP, stateP] = modelP.getEquations(state0, stateP, dt, forces, ...
                                             'iteration', itsP);
    resP = vertcat(value(problemP.equations{:}));
    if max(abs(resP)) < tol
        break
    end
    problemP = problemP.assembleSystem();
    [dxP, ~, ~] = solver.solveLinearProblem(problemP, modelP);
    for varNo = 1:numel(problemP.primaryVariables)
        name = problemP.primaryVariables{varNo};
        dx = dxP{varNo};
        switch name
            case 'pressure'
                stateP.pressure = stateP.pressure + dx;
            case 'qOs'
                for wi = 1:numel(W), stateP.wellSol(wi).qOs = stateP.wellSol(wi).qOs + dx(wi); end
            case 'qGs'
                for wi = 1:numel(W), stateP.wellSol(wi).qGs = stateP.wellSol(wi).qGs + dx(wi); end
            case 'bhp'
                for wi = 1:numel(W), stateP.wellSol(wi).bhp = stateP.wellSol(wi).bhp + dx(wi); end
        end
    end
end
stateP = modelP.computeFlash(stateP, dt, 1);

% -- transport solve ----------------------------------------------------------
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
flux_pressure = stateP.flux;
well_flux_pressure = vertcat(stateP.wellSol.flux);
well_components_pressure = vertcat(stateP.wellSol.components);
well_qOs_pressure = vertcat(stateP.wellSol.qOs);
well_qGs_pressure = vertcat(stateP.wellSol.qGs);
well_bhp_pressure = vertcat(stateP.wellSol.bhp);

sT_transport = stateT_before_update.sT;
z_transport = stateT.components;
s_transport = stateT.s;
p_transport = stateT.pressure;
well_components_transport = vertcat(stateT_before_update.wellSol.components);
well_qOs_transport = vertcat(stateT_before_update.wellSol.qOs);
well_qGs_transport = vertcat(stateT_before_update.wellSol.qGs);
well_bhp_transport = vertcat(stateT_before_update.wellSol.bhp);

inj_WI = W(1).WI;
prod_WI = W(2).WI;
inj_dZ = W(1).dZ;
prod_dZ = W(2).dZ;

save(fullfile(thisDir, 'ref_p123_comptransport_wells.mat'), '-v7', ...
     'nx', 'T0', 'p0', 'p_guess', 'z0', 'dt', ...
     'p_pressure', 'z_pressure', 'flux_pressure', ...
     'well_flux_pressure', 'well_components_pressure', ...
     'well_qOs_pressure', 'well_qGs_pressure', 'well_bhp_pressure', ...
     'sT_transport', 'z_transport', 's_transport', 'p_transport', ...
     'well_components_transport', 'well_qOs_transport', 'well_qGs_transport', ...
     'well_bhp_transport', 'inj_WI', 'prod_WI', 'inj_dZ', 'prod_dZ', ...
     'itsP', 'itsT', 'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p123_comptransport_wells.mat to %s\n', thisDir);
fprintf('  %d cells, pressure %d iterations, transport %d iterations\n', ...
        nx, itsP, itsT);
