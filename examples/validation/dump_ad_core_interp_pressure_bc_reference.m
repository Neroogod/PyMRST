% getMultiDimInterpolator and pressureBCContribADI (with bc2ADbc) in MRST
% 2024a, for validate_ad_core_interp_pressure_bc.py.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core
x = {[2; 1; 0], [10; 20; 40]};
Fx = [7 3 1; 4 2 0; 1 0.5 0.25] + [0 1 2; 0 1 2; 0 1 2];
xq = [-0.5; 0.3; 1.5; 2.7]; yq = [5; 15; 30; 45];
out = struct();
for m = {'linear', 'nearest', 'none'}
    fn = getMultiDimInterpolator(x, Fx, m{1});
    out.(m{1}) = fn(xq, yq);
end
[a, b] = initVariablesADI(xq, yq);
fn = getMultiDimInterpolator(x, Fx, 'none');
v = fn(a, b);
ad = struct('val', v.val, 'ja', full(v.jac{1}), 'jb', full(v.jac{2}));
gravity reset on
G = computeGeometry(cartGrid([3, 2, 2], [30, 20, 10]));
T_all = (1:G.faces.num)'*1e-13;
bnd = find(any(G.faces.neighbors == 0, 2));
bc = addBC([], bnd([1, 4, 9, 15]), 'pressure', [2e7; 1.5e7; 1.8e7; 2.2e7]);
bc = bc2ADbc(G, bc);
nc = G.cells.num;
[p, sw] = initVariablesADI(1.9e7 + 1e5*(1:nc)', 0.3 + 0.02*(1:nc)');
[q, cells] = pressureBCContribADI(G, struct('T_all', T_all), {p, p}, {1000 + 0*p, 800*exp(1e-9*(p - 2e7))}, ...
                                  {sw.^2/1e-3, (1 - sw).^2/5e-3}, {1 + 0*p, exp(1e-9*(p - 2e7))}, bc);
pbc = struct('q1', q{1}.val, 'q1p', full(q{1}.jac{1}), 'q1s', full(q{1}.jac{2}), ...
             'q2', q{2}.val, 'q2p', full(q{2}.jac{1}), 'q2s', full(q{2}.jac{2}), 'cells', cells, ...
             'faces', bc.face, 'map', full(bc.cell2bcface));
save(fullfile(here, 'ref_ad_core_interp_pressure_bc.mat'), '-v7', 'out', 'ad', 'pbc', 'xq', 'yq');
disp('saved');
