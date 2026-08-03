function dump_time_integration_spe1_reference()
% Deterministic two-step FIM/explicit/AIM SPE1 qualification reference.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add ad-core ad-blackoil deckformat ad-props linearsolvers test-suite
mrstDataDirectory(fullfile(mrstRoot, 'examples', 'data'));
[~, ~, ~, deck, ~] = setupSPE1();
[state0, model, schedule, nls] = initEclipseProblemAD(deck, 'TimestepStrategy', 'none');

schedule.step.val = schedule.step.val(1:2);
schedule.step.control = schedule.step.control(1:2);

modelFim = setTimeDiscretization(model, 'fully-implicit');
modelExplicit = setTimeDiscretization(model, 'explicit');
modelAim = setTimeDiscretization(model, 'adaptive-implicit');

[wsFim, statesFim, reportFim] = simulateScheduleAD(state0, modelFim, schedule, ...
    'NonLinearSolver', nls);
[wsExplicit, statesExplicit, reportExplicit] = simulateScheduleAD(state0, modelExplicit, schedule, ...
    'NonLinearSolver', nls);
[wsAim, statesAim, reportAim] = simulateScheduleAD(state0, modelAim, schedule, ...
    'NonLinearSolver', nls);

fim = localPack(statesFim, wsFim, reportFim, schedule.step.val, 'fim');
explicit = localPack(statesExplicit, wsExplicit, reportExplicit, schedule.step.val, 'explicit');
aim = localPack(statesAim, wsAim, reportAim, schedule.step.val, 'aim');
tsteps = schedule.step.val(:);

out = fullfile(fileparts(mfilename('fullpath')), 'ref_time_integration_spe1.mat');
save(out, '-v7', 'fim', 'explicit', 'aim', 'tsteps');
fprintf('Wrote %s\n', out);
end

function out = localPack(states, wellSols, report, reportSteps, mode)
    nstep = numel(states);
    nc = numel(states{1}.pressure);
    nw = numel(wellSols{1});
    out.pressure = zeros(nstep, nc);
    out.sW = zeros(nstep, nc);
    out.sG = zeros(nstep, nc);
    out.rs = zeros(nstep, nc);
    out.bhp = zeros(nstep, nw);
    out.qWs = zeros(nstep, nw);
    out.qOs = zeros(nstep, nw);
    out.qGs = zeros(nstep, nw);
    out.implicitFraction = zeros(nstep, 1);
    for k = 1:nstep
        state = states{k};
        out.pressure(k, :) = state.pressure(:)';
        out.sW(k, :) = state.s(:, 1)';
        out.sG(k, :) = state.s(:, 3)';
        out.rs(k, :) = state.rs(:)';
        out.bhp(k, :) = [wellSols{k}.bhp];
        out.qWs(k, :) = [wellSols{k}.qWs];
        out.qOs(k, :) = [wellSols{k}.qOs];
        out.qGs(k, :) = [wellSols{k}.qGs];
        if isfield(state, 'implicit')
            out.implicitFraction(k) = mean(state.implicit);
        elseif strcmp(mode, 'fim')
            out.implicitFraction(k) = 1;
        else
            out.implicitFraction(k) = 0;
        end
    end
    out.iterations = report.Iterations(:);
    out.reportSteps = reportSteps(:);
    out.ministeps = getReportMinisteps(report);
end
