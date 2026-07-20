function dump_phase115_reference()
% Phase 115 reference: compositional boundary-condition schedule.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add compositional ad-core ad-props

nx = 8;
ny = 4;
dims = [nx, ny, 1];
pdims = [1000, 1000, 10];
G = cartGrid(dims, pdims);
G = computeGeometry(G);

rng(0);
K = logNormLayers(G.cartDims);
K = K(G.cells.indexMap);
rock = makeRock(G, K*milli*darcy, 0.2);

names = {'Methane', 'CarbonDioxide', 'n-Decane'};
Tcrit = [190.5640, 304.1282, 617.7000];
Pcrit = [4599200, 7377300, 2103000];
Vcrit = [9.8628e-05, 9.4118e-05, 6.0976e-04];
acentricFactors = [0.0114, 0.2239, 0.4884];
molarMass = [0.0160, 0.0440, 0.1423];
fluid = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acentricFactors, molarMass);

minP = 75*barsa;
totTime = 20*year;
pv = poreVolume(G, rock);
bc = [];
bc = fluxside(bc, G, 'xmin', sum(pv)/totTime, 'sat', [1, 0]);
bc = pside(bc, G, 'xmax', minP, 'sat', [1, 0]);
bc.components = repmat([0.1, 0.9, 0], numel(bc.face), 1);

flowfluid = initSimpleADIFluid('rho', [1000, 500, 500], ...
    'mu', [1, 1, 1]*centi*poise, ...
    'n', [2, 2, 2], ...
    'c', [1e-5, 0, 0]/barsa);

state0 = initResSol(G, 1.5*minP, [1, 0]);
state0.T = repmat(423.15, G.cells.num, 1);
state0.components = repmat([0.3000, 0.1000, 0.6000], G.cells.num, 1);

dt = rampupTimesteps(totTime, 75*day, 12);
dt = dt(1:2);
schedule = simpleSchedule(dt, 'bc', bc);

nls = NonLinearSolver('useRelaxation', true);
modelNat = NaturalVariablesCompositionalModel(G, rock, flowfluid, fluid, 'water', false);
[~, statesNat, reportNat] = simulateScheduleAD(state0, modelNat, schedule, 'nonlinearsolver', nls);

modelOverall = OverallCompositionCompositionalModel(G, rock, flowfluid, fluid, 'water', false);
[~, statesOverall, reportOverall] = simulateScheduleAD(state0, modelOverall, schedule, 'nonlinearsolver', nls);

[overall_pressure, overall_components, overall_s] = localPack(statesOverall);
[natural_pressure, natural_components, natural_s] = localPack(statesNat);
overall_iterations = reportOverall.Iterations(:);
natural_iterations = reportNat.Iterations(:);
tsteps = dt(:);
permeability_md = K(:);
bc_faces = bc.face(:);
bc_values = bc.value(:);
bc_components = bc.components;
bc_sat = bc.sat;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p115_compositional_bc_schedule.mat');
save(out, 'overall_pressure', 'overall_components', 'overall_s', ...
    'overall_iterations', 'natural_pressure', 'natural_components', ...
    'natural_s', 'natural_iterations', 'tsteps', 'permeability_md', ...
    'bc_faces', 'bc_values', 'bc_components', 'bc_sat');
fprintf('Wrote ref_p115_compositional_bc_schedule.mat to %s\n', fileparts(out));
end

function [pressure, components, sat] = localPack(states)
    nstep = numel(states);
    nc = numel(states{1}.pressure);
    ncomp = size(states{1}.components, 2);
    nph = size(states{1}.s, 2);
    pressure = zeros(nstep, nc);
    components = zeros(nstep, nc, ncomp);
    sat = zeros(nstep, nc, nph);
    for k = 1:nstep
        st = states{k};
        pressure(k, :) = st.pressure(:)';
        components(k, :, :) = reshape(st.components, [1, nc, ncomp]);
        sat(k, :, :) = reshape(st.s, [1, nc, nph]);
    end
end
