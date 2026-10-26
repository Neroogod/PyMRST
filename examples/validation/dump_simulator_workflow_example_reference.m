% simulatorWorkflowExample in MRST 2024a without its figures and the final
% interactiveDiagnostics GUI: every stage of the workflow, for
% validate_ad_oo_simulator_workflow_example.py.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
% makeModel3's random draws (rng(0); rand, then randn), which the port takes
% as input: MATLAB's randn (Ziggurat) is not reproducible in NumPy.
rng(0); draws_a = rand(6, 1); draws_s = randn(6, 2);
% Reservoir geometry and petrophysical properties
grdecl = makeModel3([50, 50, 5], [1000, 1000, 5]*meter);
G = processGRDECL(grdecl);
G = computeGeometry(G);
[I, J, K] = gridLogicalIndices(G);
top = K < G.cartDims(3)/3;
lower = K > 2*G.cartDims(3)/3;
middle = ~(lower | top);
px = ones(G.cells.num, 1);
px(lower) = 300*milli*darcy;
px(middle) = 100*milli*darcy;
px(top) = 500*milli*darcy;
perm = [px, px, 0.1*px];
rock = makeRock(G, perm, 0.3);
% Define wells and simulation schedule
simTime = 10*year;
nstep   = 25;
refine  = 5;
pv      = poreVolume(G, rock);
injRate = 1*sum(pv)/simTime;
offset  = 10;
W = verticalWell([], G, rock, offset, offset, [],...
                'Name', 'P1', 'comp_i', [1 0], 'Val', 250*barsa, 'Type', 'bhp', 'refDepth', -8);
W = verticalWell(W, G, rock, offset, floor(G.cartDims(1)/2)+3, [],...
                'Name', 'P2', 'comp_i', [1 0], 'Val', 250*barsa, 'Type', 'bhp', 'refDepth', -8);
W = verticalWell(W, G, rock, offset, G.cartDims(2) - offset/2, [], ...
                'Name', 'P3', 'comp_i', [1 0], 'Val', 250*barsa, 'Type', 'bhp', 'refDepth', -8);
W = verticalWell(W, G, rock, G.cartDims(1)-5, offset, 1,...
                'Name', 'I1', 'comp_i', [1 0], 'Val', injRate, 'Type', 'rate', 'refDepth', -8);
startSteps = repmat((simTime/(nstep + 1))/refine, refine, 1);
restSteps =  repmat(simTime/(nstep + 1), nstep, 1);
timesteps = [startSteps; restSteps];
schedule = simpleSchedule(timesteps, 'W', W);
% Set up simulation model
fluid = initSimpleADIFluid('mu',    [1, 5]*centi*poise, ...
                           'rho',   [1000, 700]*kilogram/meter^3, ...
                           'n',     [2, 2], ...
                           'cR',    1e-8/barsa, ...
                           'phases', 'wo');
c        = 0.001/barsa;
p_ref    = 300*barsa;
fluid.bO = @(p, varargin) exp((p - p_ref)*c);
gravity reset on
model = TwoPhaseOilWaterModel(G, rock, fluid);
% Define initial state
region = getInitializationRegionsBlackOil(model, 5, 'datum_pressure', p_ref);
state0 = initStateBlackOilAD(model, region);
% Simulate base case
t = tic;
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule);
fine.walltime = toc(t);
% Upscale the model
mrstModule add coarsegrid
cdims = [5, 5, 2];
p0 = partitionUI(G, cdims);
CG = generateCoarseGrid(G, p0);
ncoarse0 = CG.cells.num;
G_fault = makeInternalBoundary(G, find(G.faces.tag > 0));
p = processPartition(G_fault, p0);
model_c = upscaleModelTPFA(model, p);
G_c    = model_c.G;
rock_c = model_c.rock;
schedule_c = upscaleSchedule(model_c, schedule);
state0_c = upscaleState(model_c, model, state0);
[wellSols_c, states_c, report_c] = simulateScheduleAD(state0_c, model_c, schedule_c);
% Flow-based transmissibility upscaling
mrstModule add incomp agglom upscaling
[~, T_c, W_c] = upscaleTrans(G_c, model.operators.T_all, ...
    'Wells', W, 'bc_method', 'wells', 'fix_trans', true);
model_c2 = upscaleModelTPFA(model, p, 'transCoarse', T_c);
schedule_c2 = schedule;
for i = 1:numel(schedule_c2.control)
    schedule_c2.control(i).W = W_c;
end
[wellSols_c2, states_c2, report_c2] = simulateScheduleAD(state0_c, model_c2, schedule_c2);
% Flow diagnostics
mrstModule add diagnostics
D   = computeTOFandTracer(states{end},   G,   rock,   'Wells', schedule.control.W);
D_c = computeTOFandTracer(states_c{end}, G_c, rock_c, 'Wells', schedule_c.control.W);
D_c2 = computeTOFandTracer(states_c2{end}, G_c, rock_c, 'Wells', schedule_c.control.W);
% Record every stage
pack = @(ws, st, rep) struct( ...
    'its', cellfun(@(r) r.Iterations, rep.ControlstepReports), ...
    'nmini', cellfun(@(r) numel(r.StepReports), rep.ControlstepReports), ...
    'p', cell2mat(cellfun(@(s) s.pressure', st, 'UniformOutput', false)), ...
    'sw', cell2mat(cellfun(@(s) s.s(:, 1)', st, 'UniformOutput', false)), ...
    'bhp', cell2mat(cellfun(@(w) [w.bhp], ws, 'UniformOutput', false)), ...
    'qWs', cell2mat(cellfun(@(w) [w.qWs], ws, 'UniformOutput', false)), ...
    'qOs', cell2mat(cellfun(@(w) [w.qOs], ws, 'UniformOutput', false)));
fine = pack(wellSols, states, report);
coarse = pack(wellSols_c, states_c, report_c);
coarse2 = pack(wellSols_c2, states_c2, report_c2);
diagpack = @(d) struct('tof', d.tof, 'itracer', d.itracer, 'ptracer', d.ptracer, ...
                       'ipart', d.ipart, 'ppart', d.ppart, 'inj', d.inj, 'prod', d.prod);
grid = struct('ncells', G.cells.num, 'nfaces', G.faces.num, 'nfault', nnz(G.faces.tag > 0), ...
              'cartDims', G.cartDims, 'perm', rock.perm, 'volumes', G.cells.volumes, ...
              'depth', G.cells.centroids(:, 3));
wells = struct('cells', {{W.cells}}, 'WI', {{W.WI}}, 'refDepth', [W.refDepth], 'val', [W.val]);
setup = struct('timesteps', timesteps, 'p0', state0.pressure, 's0', state0.s, ...
               'T', model.operators.T, 'pv', model.operators.pv);
part = struct('p0', p0, 'p', p, 'ncoarse0', ncoarse0, 'ncoarse', max(p), ...
              'nfaces_fault', G_fault.faces.num);
upscaled = struct('ncells', G_c.cells.num, 'nfaces', G_c.faces.num, 'pv', model_c.operators.pv, ...
                  'T', model_c.operators.T, 'perm', rock_c.perm, 'poro', rock_c.poro, ...
                  'p0', state0_c.pressure, 's0', state0_c.s, ...
                  'Wcells', {{schedule_c.control(1).W.cells}}, 'WI', {{schedule_c.control(1).W.WI}}, ...
                  'T2', model_c2.operators.T, 'T_c', T_c, ...
                  'W2cells', {{W_c.cells}}, 'WI2', {{W_c.WI}});
Dfine = diagpack(D); Dc = diagpack(D_c); Dc2 = diagpack(D_c2);
save(fullfile(here, 'ref_simulator_workflow_example.mat'), '-v7', 'draws_a', 'draws_s', 'grid', 'wells', 'setup', ...
     'fine', 'part', 'upscaled', 'coarse', 'coarse2', 'Dfine', 'Dc', 'Dc2');
fprintf('fine %d cells, %d its; coarse %d cells, %d its; flow-upscaled %d its\n', ...
        G.cells.num, sum(fine.its), G_c.cells.num, sum(coarse.its), sum(coarse2.its));
