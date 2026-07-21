% Dump MRST's 1D WENO face reconstruction for validate_phase348.py.
% Runs in the mrst-2024a-Geomech tree; writes ref_p348_weno.mat.
run(fullfile('..','..','..','mrst-2024a-Geomech','startup.m'));
mrstModule add ad-core ad-blackoil ad-props

N = 20;
G = computeGeometry(cartGrid([N, 1, 1], [N, 1, 1]));
rock = makeRock(G, 1, 1);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'rho', [1000, 800], 'mu', [1, 5]);
model = GenericBlackOilModel(G, rock, fluid, 'water', true, ...
                             'oil', true, 'gas', false);

weno = WENOUpwindDiscretization(model, 1);
Nconn = model.operators.N;

% Two cell-value fields: a smooth ramp and a field with a sharp front, so the
% smoothness weighting is genuinely exercised.
xc = G.cells.centroids(:, 1);
ramp  = 0.1 + 0.8 * xc / N;
front = double(xc > N/2);                 % a step (shock) at the middle

state = struct();
cases = {ramp, front};
flags = {true(size(Nconn,1),1), false(size(Nconn,1),1)};

out = struct();
for ci = 1:numel(cases)
    for fi = 1:numel(flags)
        v = weno.faceUpstream(model, state, flags{fi}, cases{ci});
        out.(sprintf('weno_c%d_f%d', ci, fi)) = full(value(v));
    end
end

ref = struct();
ref.N = Nconn;                            % 1-based interior-face neighbours
ref.cell_centroids = xc;
ref.face_centroids = G.faces.centroids(model.operators.internalConn, 1);
ref.ramp = ramp;
ref.front = front;
ref.out = out;

save(fullfile('.', 'ref_p348_weno.mat'), '-struct', 'ref');
fprintf('wrote ref_p348_weno.mat (N=%d, %d interior faces)\n', ...
        N, size(Nconn, 1));
