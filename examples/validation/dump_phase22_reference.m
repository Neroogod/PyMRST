% Phase 22 reference: the core helpers every MRST script assumes
%   pside / fluxside / boundaryFaceIndices, verticalWell,
%   getFaceTransmissibility, gaussianField, logNormLayers.
%
% The two field generators draw random numbers, and MATLAB's RNG is not
% numpy's - so the *draw* is dumped alongside the result, and the Python side
% smooths the same draw. That compares the algorithm, not the RNG.

mrstModule add ad-core
here = fileparts(mfilename('fullpath'));
ref = struct();

%% --- side-based boundary conditions -------------------------------------
G = computeGeometry(cartGrid([5, 4, 3], [50, 40, 30]));
sides = {'xmin','xmax','ymin','ymax','zmin','zmax'};
for s = 1:numel(sides)
    b = pside([], G, sides{s}, 1e5);
    ref.(['side_' sides{s}]) = sort(b.face(:));
end
% a restricted range (MATLAB is 1-based; Python will pass 0-based)
% For 'xmin' the two free logical indices are y (1..4) and z (1..3) - not x.
bsub = pside([], G, 'xmin', 1e5, [1 2], [2 3]);
ref.side_xmin_sub = sort(bsub.face(:));

bcp = pside([], G, 'xmin', 200*barsa);
bcp = pside(bcp, G, 'xmax', 100*barsa);
ref.bc_faces  = bcp.face(:);
ref.bc_values = bcp.value(:);

bcf = fluxside([], G, 'ymin', 2.0);
ref.flux_faces  = bcf.face(:);
ref.flux_values = bcf.value(:);
rawFlux = (1:numel(bcf.face))';
bcfv = fluxside([], G, 'ymin', rawFlux);
ref.flux_vector_values = bcfv.value(:);
bcrange = pside([], G, 'xmin', 3e5, 'range', 1);
ref.side_xmin_range = sort(bcrange.face(:));

%% --- vertical wells ------------------------------------------------------
rock = makeRock(G, 100*milli*darcy, 0.2);
W = verticalWell([], G, rock, 3, 2, [], 'Type','rate','Val',1.0, 'Comp_i',1);
ref.vw_cells = W(1).cells(:);
ref.vw_WI    = W(1).WI(:);
W2 = verticalWell([], G, rock, 3, 2, [1 3], 'Type','bhp','Val',1e5,'Comp_i',1);
ref.vw2_cells = W2(1).cells(:);

%% --- face transmissibility ----------------------------------------------
ref.T_face = getFaceTransmissibility(G, rock);
ref.T_half = computeTrans(G, rock);

%% --- gaussianField, on a given draw --------------------------------------
rng(0);
N2 = [30 25];
draw2 = randn(N2);
% gaussianField draws its own noise, so reproduce its body on OUR draw
sz = [7 7]; stdv = 2.0; vals = [0.1 0.3];
padSz = (sz - 1) / 2;
[xx, yy] = meshgrid(-padSz(2):padSz(2), -padSz(1):padSz(1));
D = exp(-(xx.^2 + yy.^2) / (2*stdv*stdv));  D = D / sum(D(:));
idx = cell(2,1);
for k = 1:2
    M = size(draw2, k); I = ones(1, padSz(k)); idx{k} = [I 1:M M*I];
end
gf = convn(draw2(idx{:}), D, 'valid');
mp = min(gf(:)); Mp = max(gf(:));
gf = vals(1) + (vals(2) - vals(1)) * (gf - mp) / (Mp - mp);
ref.gf_draw = draw2;
ref.gf      = gf;
ref.gf_sz   = sz;
ref.gf_std  = stdv;
ref.gf_vals = vals;

%% --- logNormLayers, on a given draw ---------------------------------------
N3 = [20 16 5];
nlay = 5;                    % one permeability layer per model layer
sigma = 2; a = 0.6; b = 2; sz3 = [9 3 3]; std3 = 4.5;
draws = zeros([N3(1:2), 3, nlay]);   % n+2 = 3 slices per single-cell layer
shifts = zeros(nlay, 1);
K = zeros(N3);
for i = 1:nlay
    d = randn([N3(1:2), 3]);
    sh = randn(1);
    draws(:,:,:,i) = d;
    shifts(i) = sh;
    k = smooth3(d - a*sh, 'gaussian', sz3, std3);
    k = exp(b + sigma*k);
    K(:,:,i) = k(:,:,2);
end
ref.ln_draws  = draws;
ref.ln_shifts = shifts;
ref.ln_K      = K(:);
ref.ln_dims   = N3(:);
ref.ln_params = [sigma; a; b; std3];
ref.ln_sz     = sz3(:);

save(fullfile(here, 'ref_p22_core_helpers.mat'), '-struct', 'ref', '-v7');
fprintf('Wrote Phase-22 reference: %d bc faces, well cells %s, K %.1f-%.1f\n', ...
        numel(ref.bc_faces), mat2str(ref.vw_cells'), min(ref.ln_K), max(ref.ln_K));
