% Phase 14 reference: Pollock streamline tracing (MRST streamlines/pollock.m)
%
% Heterogeneous quarter five-spot. We dump the converged flux together with
% the traced streamlines, so the Python side can trace from the *same* flux
% field and compare the tracer alone rather than the flow solve (which is
% already covered by the Phase 2/3 references).

mrstModule add incomp streamlines

here = fileparts(mfilename('fullpath'));

%% Grid, rock, fluid --------------------------------------------------------
n = 16;
G    = computeGeometry(cartGrid([n, n], [100, 100]));
xc   = G.cells.centroids(:,1);
yc   = G.cells.centroids(:,2);

% A smooth but strongly heterogeneous field, so the streamlines actually
% bend and the exponential branch of computeTime is exercised.
K    = (1 + 0.8*sin(2*pi*xc/50) .* cos(2*pi*yc/35)) * 100*milli*darcy;
poro = 0.15 + 0.1*cos(2*pi*xc/70) .* sin(2*pi*yc/45);
rock = makeRock(G, K, poro);

fluid = initSingleFluid('mu', 1*centi*poise, 'rho', 1000*kilogram/meter^3);
hT    = computeTrans(G, rock);

%% Wells: rate injector in cell 1, bhp producer in the far corner ----------
inj  = 1;
prod = G.cells.num;
W = addWell([], G, rock, inj,  'Type', 'rate', 'Val', 100/day, 'Comp_i', 1);
W = addWell(W,  G, rock, prod, 'Type', 'bhp',  'Val', 100*barsa, 'Comp_i', 1);

state = incompTPFA(initResSol(G, 100*barsa), G, hT, fluid, 'wells', W);

%% Trace ------------------------------------------------------------------
pv = poreVolume(G, rock);

% start positions: cells along the first row and first column, plus a few
% off-centre local coordinates to exercise the general entry point
cells  = [ (1:n)' ; ((1:n)'-1)*n + 1 ; 100; 137; 200 ];
local  = repmat([0.5 0.5], numel(cells), 1);
local(end-2:end, :) = [0.2 0.8; 0.9 0.1; 0.35 0.65];
startpos = [cells, local];

isoutflow = false(G.cells.num, 1);
isoutflow(prod) = true;

[S, T, C] = pollock(G, state, startpos, 'pvol', pv, ...
                    'isoutflow', isoutflow, 'substeps', 5);

% also trace the reverse field, which is how one finds drainage regions
[Sr, Tr, Cr] = pollock(G, state, startpos, 'pvol', pv, 'reverse', true, ...
                       'substeps', 5);

%% Pack: streamlines have different lengths, so store lengths + concatenation
pack = @(X) vertcat(X{:});
len  = cellfun(@(s) size(s,1), S);
lenr = cellfun(@(s) size(s,1), Sr);

ref = struct();
ref.n        = n;
ref.perm     = K;
ref.poro     = poro;
ref.pv       = pv;
ref.flux     = state.flux;
ref.pressure = state.pressure;
ref.startpos = startpos;
ref.inj      = inj;
ref.prod     = prod;

ref.len   = len(:);
ref.coord = pack(S);
ref.tof   = pack(T);
ref.cell  = pack(C);

ref.lenr   = lenr(:);
ref.coordr = pack(Sr);
ref.tofr   = pack(Tr);
ref.cellr  = pack(Cr);

save(fullfile(here, 'ref_p14_streamlines.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-14 streamline reference (%d lines, %d points, ' ...
         'longest TOF %.2f days)\n'], numel(S), size(ref.coord,1), ...
        max(cellfun(@(t) sum(t), T))/day);
