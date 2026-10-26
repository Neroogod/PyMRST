% blackoilTutorialSPE9 in MRST 2024a, without its figures: the quantities
% each section plots, and the simulation.
%
% The tutorial's linear solver is selectLinearSolverAD's. Without AGMG, and
% with AMGCL off (it needs MRST's MEX build, which fails on this machine),
% that is CPR with a direct pressure solve, whose GMRES does not converge in
% its 25 iterations on SPE9; the run gives up in step 1 ("Maximum number of
% substeps stopped timestep reduction"). So this records CPR's solve of the
% first linearised system (its residual, iterations and increment), and
% simulates with BackslashSolverAD instead. The tutorial's after-step
% function (getPlotAfterStep with both plots off) only shows the runtime
% panel, so it is left out; it does not change the run.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-blackoil ad-core ad-props deckformat test-suite
[G, rock, fluid, deck, state0] = setupSPE9();
model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);
model.drsMaxRel = inf;
model.dpMaxRel  = .1;
model.dsMaxAbs  = .1;
linsolve = selectLinearSolverAD(model);
lsolver = class(linsolve);
psolver = class(linsolve.ellipticSolver);
% Sections before the simulation
K = convertTo(rock.perm(:,1), milli*darcy);
[ii, jj] = gridLogicalIndices(G);
column = find(ii == 1 & jj == 1);
Rs_frac = state0.rs./model.fluid.rsSat(state0.pressure);
W = schedule.control(1).W;
sgn = [W.sign];
wno = find(strcmp({schedule.control(1).W.name}, 'PROD2'));
P = arrayfun(@(ctrl) schedule.control(ctrl).W(wno).val, schedule.step.control);
lims = schedule.control(1).W(wno).lims;
f = model.fluid;
s = (0:0.01:1)';
kr = [f.krW(s), f.krOW(s), f.krOG(s), f.krG(s)];
[x, y] = meshgrid(s);
krO = zeros(size(x));
for i = 1:size(x, 1)
    xi = x(i, :);
    yi = y(i, :);
    [~, krO(i, :), ~] = model.relPermWOG(xi, 1 - xi - yi, yi, f);
end
krO(x+y>1) = nan;
pc = [f.pcOG(s), f.pcOW(1 - s)];
pressure = (50:10:500)'*barsa;
pvt = [1./f.bW(pressure), 1./f.bG(pressure), f.pvMultR(pressure), f.muW(pressure), f.muG(pressure)];
rs = 0:25:320;
[p_g, rs_g] = meshgrid(pressure, rs);
rssat = zeros(size(p_g));
for i = 1:size(p_g, 1)
    rssat(i, :) = f.rsSat(p_g(i, :));
end
saturated = rs_g >= rssat;
rs_g(saturated) = rssat(saturated);
Bo = 1./f.bO(p_g, rs_g, saturated);
muO = f.muO(p_g, rs_g, saturated);
% CPR on the first linearised system (one-day step from state0)
dt = 1*day;
[~, forces] = model.getDrivingForces(schedule.control(1));
m1 = model.validateModel(forces);
st0 = m1.validateState(state0);
[m1, st0] = m1.updateForChangedControls(st0, forces);
[m1, st0] = m1.prepareReportstep(st0, st0, dt, forces);
[m1, st] = m1.prepareTimestep(st0, st0, dt, forces);
problem = m1.getEquations(st0, st, dt, forces, 'iteration', 1);
[~, cpr_dx, cpr_report] = linsolve.solveLinearProblem(problem, m1);
cpr_residual = cpr_report.Residual;
cpr_iterations = cpr_report.Iterations;
problem = problem.assembleSystem();
[A, b] = problem.getLinearSystem();
direct_dx = A\b;
% Simulation, with a direct linear solver (see above)
model.verbose = false;
t = tic;
[wellsols, states, reports] = simulateScheduleAD(state0, model, schedule, 'LinearSolver', BackslashSolverAD());
walltime = toc(t);
grab = @(fld) cell2mat(cellfun(@(st) st.(fld)', states, 'UniformOutput', false));
p = grab('pressure'); rs_states = grab('rs');
sat = cell2mat(cellfun(@(st) reshape(st.s, 1, []), states, 'UniformOutput', false));
[qWs, qOs, qGs, bhp] = wellSolToVector(wellsols);
its = cellfun(@(r) r.Iterations, reports.ControlstepReports);
nmini = cellfun(@(r) numel(r.StepReports), reports.ControlstepReports);
types = cellfun(@(w) strjoin({w.type}, ','), wellsols, 'UniformOutput', false);
names = {wellsols{1}.name};
% Comparison with the commercial simulator
compare = fullfile(mrstPath('ad-blackoil'), 'examples', 'spe9', 'compare');
smry = readEclipseSummaryUnFmt(fullfile(compare, 'SPE9'));
compd = 1:(size(smry.data, 2));
Tcomp = smry.get(':+:+:+:+', 'YEARS', compd);
T = convertTo(cumsum(schedule.step.val), year);
bhp_comp = [convertFrom(smry.get('PROD13', 'WBHP', compd), psia)', convertFrom(smry.get('PROD18', 'WBHP', compd), psia)'];
gas_comp = [convertFrom(smry.get('PROD13', 'WGPR', compd), 1000*ft^3/day)', convertFrom(smry.get('PROD18', 'WGPR', compd), 1000*ft^3/day)'];
bhp_mrst = [getWellOutput(wellsols, 'bhp', 'PROD13'), getWellOutput(wellsols, 'bhp', 'PROD18')];
gas_mrst = abs([getWellOutput(wellsols, 'qGs', 'PROD13'), getWellOutput(wellsols, 'qGs', 'PROD18')]);
isbhp = @(ws) arrayfun(@(x) strcmpi(x.type, 'bhp'), ws);
ctrls = cellfun(isbhp, wellsols, 'UniformOutput', false);
ctrls = double(vertcat(ctrls{:}));
save(fullfile(here, 'ref_blackoil_tutorial_spe9.mat'), 'lsolver', 'psolver', 'K', 'column', ...
     'Rs_frac', 'sgn', 'P', 'lims', 'kr', 'krO', 'pc', 'pvt', 'Bo', 'muO', 'saturated', ...
     'p', 'rs_states', 'sat', 'qWs', 'qOs', 'qGs', 'bhp', 'its', 'nmini', 'types', 'names', ...
     'Tcomp', 'T', 'bhp_comp', 'gas_comp', 'bhp_mrst', 'gas_mrst', 'ctrls', 'walltime', ...
     'cpr_dx', 'cpr_residual', 'cpr_iterations', 'direct_dx', '-v7');
