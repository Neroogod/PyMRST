%% Dump Phase 67 reference: MRST run2DCase black-oil, fixed-stress splitting.

mrstModule add ad-mechanics ad-core ad-props ad-blackoil deckformat vemmech

gravity off;

thisDir = fileparts(mfilename('fullpath'));

cartDim = [20, 4];
L = [100, 10];

G = computeGeometry(cartGrid(cartDim, L));

rock.perm = darcy*ones(G.cells.num, 1);
rock.poro = 0.3*ones(G.cells.num, 1);

fn = fullfile(ROOTDIR, 'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');
deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);
fluid = initDeckADIFluid(deck);
if isfield(fluid, 'pcOW')
    fluid = rmfield(fluid, 'pcOW');
end
if isfield(fluid, 'pcOG')
    fluid = rmfield(fluid, 'pcOG');
end
fluid.krW = @(s) s.^2;
fluid.krG = @(s) s.^2;
fluid.krOW = @(s) s.^2;
fluid.krOG = @(s) s.^2;
% This local MRST tree's SimpleWell has an OMO injector-density hook that
% dereferences fluid.rhoSinj for any injector. Stock black-oil/oil-water
% examples normally leave this unset/commented out. Gravity is off here, so
% the value is not a physics input; it only lets the modified well code run.
fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS, fluid.rhoGS];
pRef = deck.PROPS.PVTW(1);

E = 1*giga*Pascal*ones(G.cells.num, 1);
nu = 0.3*ones(G.cells.num, 1);
alpha = 1;
rock.alpha = alpha*ones(G.cells.num, 1);

bc = pside([], G, 'Ymin', 100);
bottomfaces = bc.face;
indbottom_nodes = mcolon(G.faces.nodePos(bottomfaces), ...
                         G.faces.nodePos(bottomfaces + 1) - 1);
bottom_nodes = unique(G.faces.nodes(indbottom_nodes));
disp_bc = struct('nodes', bottom_nodes, ...
                 'uu', zeros(numel(bottom_nodes), G.griddim), ...
                 'mask', ones(numel(bottom_nodes), G.griddim));

bc = pside([], G, 'Xmin', 100);
bc = pside(bc, G, 'Xmax', 100);
bc = pside(bc, G, 'Ymax', 100);
sidefaces = bc.face;
signcoef = (G.faces.neighbors(sidefaces, 1) == 0) ...
           - (G.faces.neighbors(sidefaces, 2) == 0);
n = bsxfun(@times, G.faces.normals(sidefaces, :), ...
           signcoef./G.faces.areas(sidefaces));
force_bc = struct('faces', sidefaces, 'force', bsxfun(@times, n, pRef));

mech = struct('E', E, 'nu', nu, ...
              'el_bc', struct('disp_bc', disp_bc, 'force_bc', force_bc), ...
              'load', @(x) 0*x);

model = MechFluidFixedStressSplitModel(G, rock, fluid, mech, ...
                                       'fluidModelType', 'blackoil', ...
                                       'splittingTolerance', 1e-6);
model.fluid_solver.maxIterations = 100;

W = [];
refdepth = min(G.cells.centroids(:, G.griddim));
ind = ceil(G.cartDims/2);
injcell = sub2ind(G.cartDims, ind(1), ind(2));
W = addWell(W, G, rock, injcell, ...
            'Type', 'rate', ...
            'Val', 1e2/day, ...
            'Sign', 1, ...
            'Comp_i', [0, 0, 1], ...
            'Name', 'inj', ...
            'refDepth', refdepth);

facilityModel = FacilityModel(model.fluidModel);
facilityModel = facilityModel.setupWells(W);
model.FacilityModel = facilityModel;
model.fluidModel.FacilityModel = facilityModel;

dt = [1; 5]*day;
schedule.step.val = dt;
schedule.step.control = ones(numel(dt), 1);
schedule.control = struct('W', W);

initState.pressure = pRef*ones(G.cells.num, 1);
initState.rs = 0.5*fluid.rsSat(initState.pressure);
initState.s = ones(G.cells.num, 1)*[0, 1, 0];
initState.xd = zeros(nnz(~model.mechModel.operators.isdirdofs), 1);
initState = addDerivedQuantities(model.mechModel, initState);

solver = NonLinearSolver('maxIterations', 100);
[~, states] = simulateScheduleAD(initState, model, schedule, ...
                                 'nonlinearsolver', solver);

nsteps = numel(states);
p_hist = zeros(nsteps, G.cells.num);
sw_hist = zeros(nsteps, G.cells.num);
sg_hist = zeros(nsteps, G.cells.num);
rs_hist = zeros(nsteps, G.cells.num);
xd_hist = zeros(nsteps, numel(states{1}.xd));
u_hist = zeros(nsteps, numel(states{1}.u));
vdiv_hist = zeros(nsteps, G.cells.num);
strain_hist = zeros(nsteps, G.cells.num, 3);
stress_hist = zeros(nsteps, G.cells.num, 3);
qws_hist = zeros(nsteps, 1);
qos_hist = zeros(nsteps, 1);
qgs_hist = zeros(nsteps, 1);
bhp_hist = zeros(nsteps, 1);

for i = 1:nsteps
    st = addDerivedQuantities(model.mechModel, states{i});
    p_hist(i, :) = st.pressure(:)';
    sw_hist(i, :) = st.s(:, 1)';
    sg_hist(i, :) = st.s(:, 3)';
    rs_hist(i, :) = st.rs(:)';
    xd_hist(i, :) = st.xd(:)';
    u_hist(i, :) = st.u(:)';
    vdiv_hist(i, :) = st.vdiv(:)';
    strain_hist(i, :, :) = st.strain;
    stress_hist(i, :, :) = st.stress;
    qws_hist(i) = st.wellSol(1).qWs;
    qos_hist(i) = st.wellSol(1).qOs;
    qgs_hist(i) = st.wellSol(1).qGs;
    bhp_hist(i) = st.wellSol(1).bhp;
end

save(fullfile(thisDir, 'ref_p67_run2d_blackoil_fixedstress.mat'), '-v7', ...
     'cartDim', 'L', 'pRef', 'dt', 'p_hist', 'sw_hist', 'sg_hist', ...
     'rs_hist', 'xd_hist', 'u_hist', 'vdiv_hist', 'strain_hist', ...
     'stress_hist', 'qws_hist', 'qos_hist', 'qgs_hist', 'bhp_hist');

fprintf('Wrote ref_p67_run2d_blackoil_fixedstress.mat (%d steps) to %s\n', ...
        nsteps, thisDir);
