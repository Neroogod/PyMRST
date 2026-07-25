%% Dump Phase 362 reference: AD-side upscaling (AC25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p362_upscale.mat.
%
% AC25 ports the AD-side upscaling: upscaleModelTPFA (coarse grid + coarse rock),
% upscaleState (pore-volume-weighted state coarsening) and upscaleSchedule. This
% pins the coarse *rock* and coarse *state* MRST produces against the PyMRST port
% on a small, fully deterministic fine model, so the comparison is exact.
%
% The fine rock and state are defined by closed-form patterns (no randomness) so
% PyMRST can rebuild the identical inputs and compare coarse perm, poro,
% saturation and pressure number for number.

mrstModule add ad-core ad-blackoil ad-props coarsegrid

gravity reset off;
thisDir = fileparts(mfilename('fullpath'));

% -- a deterministic 6x6 fine model, coarsened to 3x3 (each block 2x2) --------
nx = 6; ny = 6;
G = computeGeometry(cartGrid([nx, ny, 1], [600, 600, 10]*meter));
nc = G.cells.num;
i = (1:nc)';

perm = (50 + i) * milli * darcy;              % heterogeneous, monotone
poro = 0.2 + 0.001 * i;
rock = makeRock(G, perm, poro);

fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 3]*centi*poise, ...
                           'rho', [1000, 800], 'c', [0, 0]);
model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                             'oil', true, 'gas', false);
model = model.validateModel();

partition = partitionUI(G, [3, 3, 1]);        % 1-based coarse block ids
coarsemodel = upscaleModelTPFA(model, partition);

% -- coarse rock ------------------------------------------------------------
perm_c = coarsemodel.rock.perm(:, 1);
poro_c = coarsemodel.rock.poro;

% -- a deterministic fine state, then upscale it ----------------------------
sW = 0.2 + 0.01 * mod(i, 10);
state.pressure = (200 + i) * barsa;
state.s = [sW, 1 - sW];
cstate = upscaleState(coarsemodel, model, state);

s_c = cstate.s;
p_c = cstate.pressure;

save(fullfile(thisDir, 'ref_p362_upscale.mat'), '-v7', ...
     'partition', 'perm', 'poro', 'sW', 'perm_c', 'poro_c', 's_c', 'p_c');

fprintf('Wrote ref_p362_upscale.mat to %s\n', thisDir);
fprintf('  coarse cells = %d\n', coarsemodel.G.cells.num);
fprintf('  perm_c(1) = %.4f mD, poro_c(1) = %.4f\n', ...
        perm_c(1)/(milli*darcy), poro_c(1));
fprintf('  s_c(1,:) = [%.4f %.4f], p_c(1) = %.2f bar\n', ...
        s_c(1, 1), s_c(1, 2), p_c(1)/barsa);
