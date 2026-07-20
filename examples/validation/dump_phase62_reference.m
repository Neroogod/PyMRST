%% Dump Phase 62 reference: MRST run2DCase oil-water, fully coupled.
%
% This is the oil-water/fully-coupled branch of modules/ad-mechanics/examples/
% run2DCase.m, reduced to two schedule steps and with cR = 0 so the comparison
% isolates the 2D VEM + poromechanics/well coupling already present in PyMRST.

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech

gravity off;

thisDir = fileparts(mfilename('fullpath'));

cartDim = [20, 4];
L = [100, 10];
pRef = 270*barsa;

G = computeGeometry(cartGrid(cartDim, L));

rock.perm = darcy*ones(G.cells.num, 1);
rock.poro = 0.3*ones(G.cells.num, 1);

fluid = initSimpleADIFluid('phases', 'WO', ...
                           'mu', [1, 10]*centi*poise, ...
                           'n', [1, 1], ...
                           'rho', [1000, 700]*kilogram/meter^3, ...
                           'c', 1e-10*[1, 1], ...
                           'cR', 0, ...
                           'pRef', pRef);
fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS];

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

model = MechOilWaterModel(G, rock, fluid, mech, 'verbose', false);
model.toleranceCNV = 1e-8;
model.toleranceMB = 1e-10;
model.nonlinearTolerance = 1e-8;

W = [];
refdepth = min(G.cells.centroids(:, G.griddim));
ind = ceil(G.cartDims/2);
injcell = sub2ind(G.cartDims, ind(1), ind(2));
W = addWell(W, G, rock, injcell, ...
            'Type', 'rate', ...
            'Val', 1e4/day, ...
            'Sign', 1, ...
            'Comp_i', [1, 0], ...
            'Name', 'inj', ...
            'refDepth', refdepth);

facilityModel = FacilityModel(model.fluidModel);
facilityModel = facilityModel.setupWells(W);
model.FacilityModel = facilityModel;

dt = [1; 5]*day;
schedule.step.val = dt;
schedule.step.control = ones(numel(dt), 1);
schedule.control = struct('W', W);

initState.pressure = pRef*ones(G.cells.num, 1);
initState.s = ones(G.cells.num, 1)*[0, 1];
initState.xd = zeros(nnz(~model.mechModel.operators.isdirdofs), 1);
initState = addDerivedQuantities(model.mechModel, initState);

solver = NonLinearSolver('maxIterations', 100);
[~, states] = simulateScheduleAD(initState, model, schedule, ...
                                 'nonlinearsolver', solver);

nsteps = numel(states);
p_hist = zeros(nsteps, G.cells.num);
sw_hist = zeros(nsteps, G.cells.num);
xd_hist = zeros(nsteps, numel(states{1}.xd));
u_hist = zeros(nsteps, numel(states{1}.u));
vdiv_hist = zeros(nsteps, G.cells.num);
strain_hist = zeros(nsteps, G.cells.num, 3);
stress_hist = zeros(nsteps, G.cells.num, 3);
qws_hist = zeros(nsteps, 1);
qos_hist = zeros(nsteps, 1);
bhp_hist = zeros(nsteps, 1);

for i = 1:nsteps
    st = addDerivedQuantities(model.mechModel, states{i});
    p_hist(i, :) = st.pressure(:)';
    sw_hist(i, :) = st.s(:, 1)';
    xd_hist(i, :) = st.xd(:)';
    u_hist(i, :) = st.u(:)';
    vdiv_hist(i, :) = st.vdiv(:)';
    strain_hist(i, :, :) = st.strain;
    stress_hist(i, :, :) = st.stress;
    qws_hist(i) = st.wellSol(1).qWs;
    qos_hist(i) = st.wellSol(1).qOs;
    bhp_hist(i) = st.wellSol(1).bhp;
end

save(fullfile(thisDir, 'ref_p62_run2d_oilwater.mat'), '-v7', ...
     'cartDim', 'L', 'pRef', 'dt', 'p_hist', 'sw_hist', 'xd_hist', ...
     'u_hist', 'vdiv_hist', 'strain_hist', 'stress_hist', ...
     'qws_hist', 'qos_hist', 'bhp_hist');

fprintf('Wrote ref_p62_run2d_oilwater.mat (%d steps) to %s\n', nsteps, thisDir);
