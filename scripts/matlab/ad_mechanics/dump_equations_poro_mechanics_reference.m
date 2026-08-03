function dump_equations_poro_mechanics_reference(output_file)
% Direct residual/Jacobian fixture for equationsPoroMechanics.m.
if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_equations_poro_mechanics.mat');
end
here = fileparts(mfilename('fullpath'));
pymrstRoot = fileparts(fileparts(fileparts(here)));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add ad-core ad-mechanics

model.G.cells.volumes = [2; 4];
model.operators.A = sparse([4, -1, 0; -1, 3, -0.5; 0, -0.5, 2]);
model.operators.gradP = sparse([1, 0.2; -0.3, 0.8; 0.5, -0.4]);
model.operators.rhs = [0.7; -1.2; 0.4];
model.rock.alpha = [0.75; 0.9];
xd0 = [0.1; -0.2; 0.3];
p0 = [12e6; 15e6];
[xd, p] = initVariablesADI(xd0, p0);
[eqs, names, types] = equationsPoroMechanics(xd, model, p);
eq = eqs{1};
residual = eq.val;
jac_xd = full(eq.jac{1});
jac_p = full(eq.jac{2});
save(output_file, 'xd0', 'p0', 'residual', 'jac_xd', 'jac_p', ...
    'names', 'types', 'model', '-v7');
fprintf('Wrote %s\n', output_file);
end
