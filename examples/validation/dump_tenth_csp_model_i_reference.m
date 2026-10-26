% tenthCSP_Model_I in MRST 2024a without its figures (and without its
% after-step plot function, which only plots): the setup and the full run
% (validate_ad_oo_tenth_csp_model_i.py).
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props spe10
gravity reset on
cartDims = [ 100, 1, 20 ];
physDims = cartDims .* [ 25, 25, 2.5 ]*ft;
G = computeGeometry(cartGrid(cartDims, physDims));
rock = getSPE10_model_1_rock();
IJK = gridLogicalIndices(G);
I   = find(IJK{1}(:,1) == 1);
P   = find(IJK{1}(:,1) == G.cartDims(1));
W = addWell([], G, rock, I, 'Comp_i', [ 0, 1 ], 'Type', 'rate', ...
            'Val', 6.97*meter^3/day, 'Radius', (1/2)*ft, 'Dir', 'z', ...
            'Sign', +1, 'Name', 'I', 'refDepth', 0*ft);
W = addWell(W, G, rock, P, 'Comp_i', [ 1, 1 ], 'Type', 'bhp', ...
            'Val', 95*psia, 'Radius', (1/2)*ft, 'Dir', 'z', ...
            'Sign', -1, 'Name', 'P', 'refDepth', 0*ft);
kr_deck = getSPE10_model_1_relperm();
fluid = initSimpleADIFluid('mu'    , [  1, 0.01]*centi*poise, ...
                           'rho'   , [700, 1   ]*kilogram/meter^3, ...
                           'cR'    , 6.0e-5/barsa, ...
                           'phases', 'OG');
fluid_kr = assignSGOF(fluid, kr_deck.PROPS.SGOF, struct('sat', 1, ...
                                               'interp1d', @interpTable));
fluid.krG = fluid_kr.krG{1};
fluid.krO = fluid_kr.krOG{1};
model = GenericBlackOilModel(G, rock, fluid, 'gravity', gravity, 'disgas', false,...
    'vapoil', false, 'water', false, 'oil', true, 'gas', true);
region = getInitializationRegionsBlackOil(model, 0, 'datum_pressure', 100*psia);
state0 = initStateBlackOilAD(model, region);
timesteps = [ 0.1, 0.2, 0.3, 0.4, repmat(0.5, [1, 6]), ones([1, 6]), ...
              repmat(2, [1, 5]), repmat(5, [1, 8]), repmat(10, [1, 9]), ...
              repmat(20, [1, 5]), repmat(30, [1, 50]), ...
              repmat(50, [1, 38]) ]*day;
schedule = simpleSchedule(timesteps, 'W', W);
t = tic;
[wellSols, states, report] = simulateScheduleAD(state0, model, schedule);
walltime = toc(t);
s_ = linspace(0, 1, 101)';
krG = fluid.krG(s_); krO = fluid.krO(s_); pvMultR = fluid.pvMultR((50:10:200)'*barsa);
perm = rock.perm; poro = rock.poro;
WI_I = W(1).WI; WI_P = W(2).WI; cells_I = W(1).cells; cells_P = W(2).cells;
p0 = state0.pressure; s0 = state0.s;
its = cellfun(@(r) r.Iterations, report.ControlstepReports);
nmini = cellfun(@(r) numel(r.StepReports), report.ControlstepReports);
p = cell2mat(cellfun(@(st) st.pressure', states, 'UniformOutput', false));
sg = cell2mat(cellfun(@(st) st.s(:, 2)', states, 'UniformOutput', false));
bhp = cell2mat(cellfun(@(w) [w.bhp], wellSols, 'UniformOutput', false));
qOs = cell2mat(cellfun(@(w) [w.qOs], wellSols, 'UniformOutput', false));
qGs = cell2mat(cellfun(@(w) [w.qGs], wellSols, 'UniformOutput', false));
types = cellfun(@(w) strjoin({w.type}, ','), wellSols, 'UniformOutput', false);
s2 = states{2}.s;   % step 2's saturations, exact zeros included
save(fullfile(here, 'ref_tenth_csp_model_i.mat'), '-v7', 'perm', 'poro', 'krG', 'krO', ...
     'pvMultR', 'WI_I', 'WI_P', 'cells_I', 'cells_P', 'p0', 's0', 'its', 'nmini', 'p', 'sg', ...
     'bhp', 'qOs', 'qGs', 'types', 's2', 'walltime');
fprintf('tenthCSP_Model_I: %d steps, %d iterations, %.0f s\n', numel(its), sum(its), walltime);
