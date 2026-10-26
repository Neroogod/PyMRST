% LinearizedProblem in MRST 2024a on the first Newton system of a small
% two-well oil-water GenericBlackOilModel, for
% validate_ad_core_linearized_problem.py.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
gravity reset on
G = computeGeometry(cartGrid([4, 3, 2], [400, 300, 20]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 4]*centi*poise, 'rho', [1000, 800], ...
                           'n', [2, 2], 'c', [1e-9, 1e-8]/barsa);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
W = addWell([], G, rock, [1; 13], 'type', 'rate', 'val', 50/day, 'comp_i', [1, 0], 'name', 'I');
W = addWell(W, G, rock, [12; 24], 'type', 'bhp', 'val', 150*barsa, 'comp_i', [1, 0], 'name', 'P');
state0 = initResSol(G, 200*barsa, [0.2, 0.8]);
state0.wellSol = initWellSolAD(W, model, state0);
dt = 10*day;
forces = model.getValidDrivingForces(); forces.W = W;
model = model.validateModel(forces);
state = model.validateState(state0);
[model, state] = model.prepareTimestep(state, state0, dt, forces);
problem = model.getEquations(state0, state, dt, forces, 'iteration', 1);
[A, b] = problem.getLinearSystem();
info = struct('names', {problem.equationNames}, 'types', {problem.types}, ...
              'vars', {problem.primaryVariables}, 'norm2', problem.norm(), 'normInf', problem.norm(inf));
perm = numel(problem.equations):-1:1;
[Ar, br] = problem.reorderEquations(perm).getLinearSystem();
wellName = problem.equationNames{find(~problem.indexOfType('cell'), 1)};
[pe, elim] = problem.eliminateVariable(wellName);
[Ae, be] = pe.getLinearSystem();
[pc, eliminated] = problem.reduceToSingleVariableType('cell');
[Ac, bc] = pc.getLinearSystem();
xc = Ac\bc;
nv = cellfun(@(j) size(j, 2), pc.equations{1}.jac);
dxc = mat2cell(xc, nv, 1);
dx = pc.recoverFromSingleVariableType(problem, dxc, eliminated);
dxfull = vertcat(dx{:});
[dvar, rest, popped] = problem.popIncrement(dx, problem.primaryVariables{1});
save(fullfile(here, 'ref_ad_core_linearized_problem.mat'), '-v7', 'A', 'b', 'info', 'perm', 'Ar', 'br', ...
     'wellName', 'Ae', 'be', 'Ac', 'bc', 'dxfull', 'dvar', 'popped');
fprintf('saved: %d equations, %d unknowns, cell system %d\n', numel(problem.equations), size(A, 1), size(Ac, 1));
