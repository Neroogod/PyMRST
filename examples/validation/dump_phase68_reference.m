%% Dump Phase 68 reference: VEM stress and stress eigen utilities.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

%% 2D
nx2 = 3; ny2 = 2;
L2 = [3, 2];
E2 = 1e9;
nu2 = 0.25;

G2 = computeGeometry(cartGrid([nx2, ny2], L2));
G2 = createAugmentedGrid(G2);
C2 = Enu2C(E2*ones(G2.cells.num, 1), nu2*ones(G2.cells.num, 1), G2);
[~, op2] = VEM_assemble(G2, C2);

x2 = G2.nodes.coords(:, 1);
y2 = G2.nodes.coords(:, 2);
uu2 = [0.01*x2 + 0.02*y2, -0.015*x2 + 0.005*y2];
[stress2, strain2] = calculateStressVEM(G2, uu2, op2);
[sigm2, evec2] = calStressEigsVEM(G2, stress2);

%% 3D
nx3 = 2; ny3 = 2; nz3 = 2;
L3 = [2, 2, 2];
E3 = 1.2e9;
nu3 = 0.22;

G3 = computeGeometry(cartGrid([nx3, ny3, nz3], L3));
G3 = createAugmentedGrid(G3);
C3 = Enu2C(E3*ones(G3.cells.num, 1), nu3*ones(G3.cells.num, 1), G3);
[~, op3] = VEM_assemble(G3, C3);

x3 = G3.nodes.coords(:, 1);
y3 = G3.nodes.coords(:, 2);
z3 = G3.nodes.coords(:, 3);
uu3 = [0.010*x3 + 0.003*y3 - 0.002*z3, ...
       0.004*x3 - 0.006*y3 + 0.005*z3, ...
      -0.001*x3 + 0.002*y3 + 0.008*z3];
[stress3, strain3] = calculateStressVEM(G3, uu3, op3);
[sigm3, evec3] = calStressEigsVEM(G3, stress3);

save(fullfile(thisDir, 'ref_p68_vem_stress.mat'), '-v7', ...
     'nx2', 'ny2', 'L2', 'E2', 'nu2', 'uu2', 'stress2', 'strain2', ...
     'sigm2', 'evec2', ...
     'nx3', 'ny3', 'nz3', 'L3', 'E3', 'nu3', 'uu3', 'stress3', ...
     'strain3', 'sigm3', 'evec3');

fprintf('Wrote ref_p68_vem_stress.mat to %s\n', thisDir);
