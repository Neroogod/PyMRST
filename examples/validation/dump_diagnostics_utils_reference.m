% The computations behind interactiveDiagnostics, on the final state of
% simulatorWorkflowExample's fine model (MRST 2024a): computeTOFandTracer with
% the GUI's options plus well TOFs and first arrival, computeWellPairs,
% computeFandPhi, computeLorenz and computeSweep; matrix inputs and synthetic
% RTDs for computeFandPhiFromDist. For validate_diagnostics_utils.py.
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
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule);
fprintf('fine run: %d iterations\n', sum(cellfun(@(r) r.Iterations, report.ControlstepReports)));
mrstModule add diagnostics
st = states{end};
% interactiveDiagnostics' computeTOFAndTracerAndWellPairs, with well TOFs and
% first arrival switched on
D = computeTOFandTracer(st, G, rock, 'wells', W, 'processCycles', true, ...
                        'computeWellTOFs', true, 'firstArrival', true);
WP = computeWellPairs(st, G, rock, W, D);
pv = poreVolume(G, rock);
[F, Phi] = computeFandPhi(pv, D.tof);
Lc = computeLorenz(F, Phi);
[Ev, tD] = computeSweep(F, Phi);
% Matrix inputs: Lorenz per column, computeSweep's linear indexing
F2 = [F, F.^2]; Phi2 = [Phi, sqrt(Phi)];
Lc2 = computeLorenz(F2, Phi2);
[Ev2, tD2] = computeSweep(F2, Phi2);
% computeFandPhiFromDist on synthetic RTDs
t = linspace(0, 10, 41)' * [1, 2];
t(3, 2) = NaN;
RTD = struct('creator', 'computeRTD', 't', t, 'values', exp(-t) .* [1, 0.5], ...
             'volumes', [3, 4], 'allocations', [1, 2], 'injectorFlux', [5; 6], ...
             'pairIx', [1, 1; 2, 1]);
[Fr, Phir] = computeFandPhi(RTD);
RTDs = RTD; RTDs.t = linspace(0, 10, 41)' * [1, 1];
[Frs, Phirs] = computeFandPhiFromDist(RTDs, 'sum', true);
RTDe = RTD; RTDe.creator = 'estimateRTD';
[Fre, Phire] = computeFandPhi(RTDe);
Dsave = struct('tof', D.tof, 'itracer', D.itracer, 'ptracer', D.ptracer, 'ipart', D.ipart, ...
               'ppart', D.ppart, 'inj', D.inj, 'prod', D.prod, 'itof', D.itof, 'ptof', D.ptof, ...
               'ifa', D.ifa, 'pfa', D.pfa);
WPsave = struct('pairs', {WP.pairs}, 'vols', WP.vols, 'pairIx', WP.pairIx, ...
                'inj_alloc', {{WP.inj.alloc}}, 'inj_ralloc', {{WP.inj.ralloc}}, 'inj_z', {{WP.inj.z}}, ...
                'prod_alloc', {{WP.prod.alloc}}, 'prod_ralloc', {{WP.prod.ralloc}}, 'prod_z', {{WP.prod.z}});
save(fullfile(here, 'ref_diagnostics_utils.mat'), '-v7', 'draws_a', 'draws_s', 'Dsave', 'WPsave', ...
     'F', 'Phi', 'Lc', 'Ev', 'tD', 'F2', 'Phi2', 'Lc2', 'Ev2', 'tD2', 'RTD', 'Fr', 'Phir', ...
     'RTDs', 'Frs', 'Phirs', 'Fre', 'Phire');
fprintf('saved: Lc %.6f, %d well pairs\n', Lc, numel(WP.pairs));
