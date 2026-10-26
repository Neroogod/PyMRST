% blackoilTutorialOnePhase, as published, without the plots; saves results.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-props ad-core ad-blackoil
G    = computeGeometry(cartGrid([50 50],[1000 100]));
rock = makeRock(G, 100*milli*darcy, 0.3);
pR  = 200*barsa;
fluid = initSimpleADIFluid('phases','W', 'mu', 1*centi*poise, 'rho', 1000, ...
                           'c', 1e-4/barsa, 'cR', 1e-5/barsa);
gravity reset on
wModel = WaterModel(G, rock, fluid,'gravity',[0 -norm(gravity)]);
wModel = wModel.validateModel();
wc = sub2ind(G.cartDims, floor(G.cartDims(1)/2), G.cartDims(2));
W = addWell([], G, rock,  wc, 'Type', 'bhp', 'Val', pR-50*barsa, ...
        'Radius', 0.1, 'Name', 'P1','Comp_i',1,'sign',1);
bc=pside([],G,'South',200*barsa,'sat',1);
schedule1 = simpleSchedule(diff(linspace(0,5*day,41)), 'bc', bc, 'W', W);
schedule2 = simpleSchedule(diff(linspace(0,5*day,41)), 'W', W);
state = initResSol(G, pR);
state = wModel.validateState(state);
nonlinear = NonLinearSolver();
state = nonlinear.solveTimestep(state, 10000*day, wModel, 'bc', bc);
[wellSols1, states1, rep1] = simulateScheduleAD(state, wModel, schedule1);
[wellSols2, states2, rep2] = simulateScheduleAD(state, wModel, schedule2);
p0 = state.pressure;
p1 = cell2mat(cellfun(@(s) s.pressure', states1, 'UniformOutput', false));
p2 = cell2mat(cellfun(@(s) s.pressure', states2, 'UniformOutput', false));
q1 = cellfun(@(w) w(1).qWs, wellSols1);
q2 = cellfun(@(w) w(1).qWs, wellSols2);
its1 = rep1.Iterations; its2 = rep2.Iterations;
save(fullfile(here, 'ref_ad_oo_one_phase.mat'), ...
     'p0', 'p1', 'p2', 'q1', 'q2', 'its1', 'its2', 'wc', '-v7');
