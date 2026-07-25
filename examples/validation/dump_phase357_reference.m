%% Dump Phase 357 reference: saturation CFL on a Buckley-Leverett column, MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p357_cfl.mat.
%
% AC21 ports `estimateSaturationCFL` and the two-phase branch of
% `getFractionalFlowMagnitude` to `ad_core/utils/cfl.py`. This pins the port
% against MRST on the case the CFL is for: a 1D Buckley-Leverett water/oil
% column, quadratic relperm, a saturation front, and a known through-flux.
%
% The saturation profile and the interior-face phase flux are *fixed here* and
% dumped, so PyMRST feeds the identical inputs to its kernel and the comparison
% isolates the CFL arithmetic from any difference in how a flux is assembled.
% What is saved: the fractional-flow magnitude F (cell), the CFL (cell), and the
% ingredients (sw, interior flux v, neighbour list N, pv, dt).

mrstModule add ad-core ad-props ad-blackoil

gravity reset off;
thisDir = fileparts(mfilename('fullpath'));

% -- a 1D water/oil column, quadratic relperm, equal viscosity ---------------
nx = 20;
G = computeGeometry(cartGrid([nx, 1, 1], [200, 10, 10]*meter));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 1]*centi*poise, ...
                           'rho', [1000, 1000], 'c', [0, 0]);
model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                             'oil', true, 'gas', false);
model = model.validateModel();

% -- a Buckley-Leverett state: a saturation front, a known through-flux -------
nc = G.cells.num;
sw = 0.8 - 0.6*((0:nc-1)'/(nc-1));       % 0.8 at the inlet down to 0.2
state.pressure = repmat(200*barsa, nc, 1);
state.s = [sw, 1 - sw];

% interior-face phase flux: a total that tapers along the column, split by a
% nominal fractional flow so the two phase columns are non-trivial (only the
% sum enters the CFL, but dump both to exercise the sum-and-sign logic)
nf = size(model.operators.N, 1);
qT = (1e-6)*(1 - 0.3*((0:nf-1)'/(nf-1)));   % all positive: left-to-right
v_int = [0.6*qT, 0.4*qT];

% write it into the full-face flux array estimateSaturationCFL reads
flux = zeros(G.faces.num, 2);
flux(model.operators.internalConn, :) = v_int;
state.flux = flux;

% -- MRST's own CFL and fractional-flow magnitude ----------------------------
dt = 20*day;
cfl = estimateSaturationCFL(model, state, dt);
[F, ~] = getFractionalFlowMagnitude(model, state);

N  = model.operators.N;                  % 1-based neighbour list
pv = model.operators.pv;

save(fullfile(thisDir, 'ref_p357_cfl.mat'), '-v7', ...
     'sw', 'v_int', 'N', 'pv', 'dt', 'F', 'cfl');

fprintf('Wrote ref_p357_cfl.mat to %s\n', thisDir);
fprintf('  nc = %d, interior faces = %d\n', nc, nf);
fprintf('  max CFL = %.6e,  max F = %.6e\n', max(cfl), max(F));
