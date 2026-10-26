% Reference for equationsBlackOil (validate_equations_black_oil.py): SPE9 as a
% ThreePhaseBlackOilModel with dissolved gas and its first control's wells,
% on a state that mixes undersaturated (status 1) and three-phase (status 3)
% cells and perturbs the pressure. The residuals and the assembled system.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat test-suite
gravity reset on
[G, rock, fluid, deck, state0] = setupSPE9();
model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true, 'inputdata', deck);
schedule = convertDeckScheduleToMRST(model, deck);
[~, forces] = model.getDrivingForces(schedule.control(1));
model = model.validateModel(forces);
st0 = model.validateState(state0);
[model, st0] = model.updateForChangedControls(st0, forces);
nc = G.cells.num;
st = st0;
gas = (1:7:nc)';
st.s(gas, 3) = 0.05;
st.s(gas, 2) = 1 - st.s(gas, 1) - st.s(gas, 3);
st.pressure = st.pressure.*(1 + 0.01*sin((1:nc)'));
st.rs(gas) = fluid.rsSat(st.pressure(gas));
dt = 1*day;
[model, st] = model.prepareReportstep(st, st0, dt, forces);
[model, st] = model.prepareTimestep(st, st0, dt, forces);
problem = model.getEquations(st0, st, dt, forces, 'iteration', 1);
eqnames = problem.equationNames;
res = cellfun(@(e) value(e), problem.equations, 'UniformOutput', false);
problem = problem.assembleSystem();
[A, b] = problem.getLinearSystem();
p0 = st0.pressure; s0 = st0.s; rs0 = st0.rs;
p = st.pressure; s = st.s; rs = st.rs;
bhp = [st.wellSol.bhp]; qWs = [st.wellSol.qWs]; qOs = [st.wellSol.qOs]; qGs = [st.wellSol.qGs];
save(fullfile(here, 'ref_equations_black_oil.mat'), '-v7', 'eqnames', 'res', 'A', 'b', ...
     'p0', 's0', 'rs0', 'p', 's', 'rs', 'bhp', 'qWs', 'qOs', 'qGs');
