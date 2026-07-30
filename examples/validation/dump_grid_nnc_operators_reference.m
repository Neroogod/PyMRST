%% Reference data for setupOperatorsTPFA with explicit NNCs.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core
G = computeGeometry(cartGrid([3, 1, 1], [3, 1, 1]));
G.nnc.cells = [1, 3];
G.nnc.trans = 7.5;
rock = makeRock(G, ones(G.cells.num, 1), 0.2);
op = setupOperatorsTPFA(G, rock);

neighbors = op.N;
transmissibility = op.T;
gradient = op.Grad([1; 4; 9]);
divergence = op.Div([0; 0; 2]);

save(fullfile(thisDir, 'ref_grid_nnc_operators.mat'), '-v7', ...
     'neighbors', 'transmissibility', 'gradient', 'divergence');
fprintf('Wrote NNC operator reference to %s\n', thisDir);
