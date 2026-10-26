% setupOperatorsTPFA and faceUpstr in MRST 2024a, for
% validate_ad_core_operators.py: every field and operator on a heterogeneous
% grid, and the transmissibility and NNC branches.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core
G = computeGeometry(cartGrid([4, 3, 2], [40, 30, 20]));
nc = G.cells.num;
perm = (1 + (1:nc)') * 1e-13;
rock = makeRock(G, [perm, 2*perm, 0.5*perm], 0.1 + 0.01*(1:nc)');
rock.ntg = 0.5 + 0.02*(1:nc)';
op = setupOperatorsTPFA(G, rock);
nf = size(op.N, 1);
x = sin(1:nc)';
f = cos(1:nf)';
flag = mod(1:nf, 3)' == 0;
base = struct('T', op.T, 'T_all', op.T_all, 'N', op.N, 'internalConn', op.internalConn, ...
              'pv', op.pv, 'C', full(op.C), 'M', full(op.M), 'grad', op.Grad(x), ...
              'div', op.Div(f), 'accdiv', op.AccDiv(x, f), 'avg', op.faceAvg(x), ...
              'upw', op.faceUpstr(flag, x), 'upw_true', op.faceUpstr(true, x), ...
              'upw_matrix', op.faceUpstr(flag, [x, 2*x]));
G2 = G;
G2.nnc = struct('cells', [1, 24; 2, 1; 0, 5; 7, 18], 'trans', [5; 9; 7; 3]);
opn = setupOperatorsTPFA(G2, rock);
nnc = struct('N', opn.N, 'T', opn.T, 'T_all', opn.T_all, 'internalConn', opn.internalConn);
Tint = (1:nnz(all(G.faces.neighbors > 0, 2)))';
opi = setupOperatorsTPFA(G, rock, 'trans', Tint);
Tall = (1:G.faces.num)' / 10;
opa = setupOperatorsTPFA(G, rock, 'trans', Tall);
pairs = [1, 2; 2, 3; 1, 5];
ope = setupOperatorsTPFA(G, rock, 'neighbors', pairs, 'trans', [4; 5; 6]);
G3 = G; G3.faces.TRANS = (1:G.faces.num)' * 2;
opt = setupOperatorsTPFA(G3, rock);
opp = setupOperatorsTPFA(G, rock, 'porv', (1:nc)');
given = struct('T_int', opi.T, 'T_all_int', opi.T_all, 'T_full', opa.T, 'T_all_full', opa.T_all, ...
               'N_pairs', ope.N, 'T_pairs', ope.T, 'T_all_pairs', ope.T_all, ...
               'conn_pairs', ope.internalConn, 'grad_pairs', ope.Grad(x), ...
               'T_grid', opt.T, 'pv_porv', opp.pv);
xu = faceUpstr(flag, [x, x.^2], op.N, [nf, nc]);
[xu_scalar, xc] = faceUpstr(false, x, op.N, [nf, nc]);
upstr = struct('xu', xu, 'xu_scalar', xu_scalar, 'xc', xc);
save(fullfile(here, 'ref_ad_core_operators.mat'), '-v7', 'base', 'nnc', 'given', 'upstr', 'x', 'f', 'flag');
fprintf('saved: %d connections, %d with NNCs\n', nf, size(opn.N, 1));
