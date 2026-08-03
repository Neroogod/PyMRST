function dump_custom_vem_stabilization_reference(output_file)
% Direct setupOperatorsVEM fixture for MRST's experimental S option.
if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_custom_vem_stabilization.mat');
end
mrstModule add vemmech ad-mechanics
G = computeGeometry(cartGrid([1, 1], [1, 1]));
G = createAugmentedGrid(G);
C = Enu2C(1e9, 0.25, G);
nodes = (1:G.nodes.num)';
el_bc.disp_bc.nodes = nodes;
el_bc.disp_bc.uu = zeros(G.nodes.num, 2);
el_bc.disp_bc.mask = repmat([true, false], G.nodes.num, 1);
el_bc.force_bc = [];
load = @(x) repmat([2, -3], size(x, 1), 1);
custom_S = 7.5e8 * speye(G.griddim * G.nodes.num);
operators = setupOperatorsVEM(G, C, el_bc, load, 1, custom_S);
A = full(operators.A);
rhs = operators.rhs;
strain = full(operators.strain);
stress = full(operators.stress);
isdirdofs = operators.isdirdofs;
save(output_file, 'A', 'rhs', 'strain', 'stress', 'isdirdofs', '-v7');
fprintf('Wrote %s\n', output_file);
end
