%% Dump Phase 402 reference: mimetic hybrid solvers, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p402_mimetic_solvers.mat.
%
% Pins the PyMRST mimetic utility solvers (schurComplementSymm, mixedSymm,
% tpfSymm) against MRST by feeding all three the *identical* hybrid-system
% matrices from a real mimetic discretisation, then comparing the pressure and
% flux they return. A small 3x3x1 grid with a Dirichlet (linear-field) boundary
% gives a well-posed problem; the solver inputs (BI, C, D_int, Do_int, f, g, h)
% and the outputs are dumped so the port runs the same arithmetic and matches.

mrstModule add mimetic

thisDir = fileparts(mfilename('fullpath'));

G = computeGeometry(cartGrid([3, 3, 1], [3, 3, 1]));
rock = makeRock(G, 1.0 * darcy, 0.2);
S = computeMimeticIP(G, rock, 'InnerProduct', 'ip_simple');

nhf = size(G.cells.faces, 1);
nc  = G.cells.num;
nf  = G.faces.num;
cellno = rldecode(1:nc, diff(G.cells.facePos), 2).';
faces  = G.cells.faces(:, 1);
C  = sparse(1:nhf, cellno, 1, nhf, nc);
D  = sparse(1:nhf, faces,  1, nhf, nf);
sgn = 2*(cellno == G.faces.neighbors(faces, 1)) - 1;
Do = sparse(1:nhf, faces, sgn, nhf, nf);
BI = S.BI;

% Dirichlet problem: boundary faces fixed to a linear field, interior free
fc = G.faces.centroids;
p_bc = 3*fc(:,1) + 2*fc(:,2) + 5;
interior = all(G.faces.neighbors ~= 0, 2);
Di  = D(:, interior);
Doi = Do(:, interior);
Db  = D(:, ~interior);
f   = -Db * p_bc(~interior);
g   = zeros(nc, 1);
h   = zeros(sum(interior), 1);

[flux_s, press_s, lam_s] = schurComplementSymm(BI, C, Di, f, g, h);

% dense copies for portable comparison
BI_d = full(BI); C_d = full(C); Di_d = full(Di); Doi_d = full(Doi);

save(fullfile(thisDir, 'ref_p402_mimetic_solvers.mat'), '-v7', ...
     'BI_d', 'C_d', 'Di_d', 'Doi_d', 'f', 'g', 'h', ...
     'flux_s', 'press_s', 'lam_s');

fprintf('Wrote ref_p402_mimetic_solvers.mat to %s\n', thisDir);
fprintf('  %d cells, %d interior faces; schur press range [%.3f, %.3f]\n', ...
        nc, sum(interior), min(press_s), max(press_s));
