% blackoilTutorialGravSeg, as published, without the plots; saves results.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
dims = [5 5 10];
G = cartGrid(dims, [1, 1, 5]*meter);
G = computeGeometry(G);
rock = makeRock(G, 100*milli*darcy, 0.5);
fluid = initSimpleADIFluid('phases', 'WO', 'mu', [1, 10]*centi*poise, ...
                           'n',  [1, 1], 'rho', [1000, 700]*kilogram/meter^3);
gravity reset on
model = TwoPhaseOilWaterModel(G, rock, fluid);
[ii, jj, kk] = gridLogicalIndices(G);
lower = kk > 5;
sW = ones(G.cells.num, 1);
sW(lower) = 0;
s = [sW, 1 - sW];
state = initResSol(G, 100*barsa, s);
bc = [];
bc = pside(bc, G, 'ZMin', 100*barsa, 'sat', [0 1]);
dt = repmat(90*day, 20, 1);
schedule = simpleSchedule(dt, 'bc', bc);
[wellSols, states, rep] = simulateScheduleAD(state, model, schedule);
p = cell2mat(cellfun(@(s) s.pressure', states, 'UniformOutput', false));
sw = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
its = rep.Iterations;
save(fullfile(here, 'ref_ad_oo_grav_seg.mat'), ...
     'p', 'sw', 'its', '-v7');
