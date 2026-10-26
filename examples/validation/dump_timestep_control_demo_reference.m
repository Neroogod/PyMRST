% timestepControlDemo in MRST 2024a, without its figures: the five runs'
% ministep lengths, well curves and Newton iteration totals.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-props deckformat ad-core ad-blackoil test-suite
[G, rock, fluid, deck, state] = setupSPE1();
model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);
[wellSols, states, report] = simulateScheduleAD(state, model, schedule);
schedule_small = compressSchedule(schedule);
rampup = 1*day;
targetIts = [4 8 15 25];
[reports, ws] = deal(cell(numel(targetIts), 1));
for i = 1:numel(targetIts)
    timestepper = ...
       IterationCountTimeStepSelector('targetIterationCount', targetIts(i),...
                                      'minRelativeAdjustment', sqrt(eps),...
                                      'maxRelativeAdjustment', 4, ...
                                      'firstRampupStep',       rampup, ...
                                      'verbose', true);
    nonlinear = NonLinearSolver('timeStepSelector', timestepper, ...
                                'maxiterations', 4*targetIts(i));
    [ws{i}, ~, reports{i}] = simulateScheduleAD(state, model, schedule_small,...
                        'nonlinearSolver', nonlinear, 'outputMinisteps', true);
end
wsols = vertcat({wellSols}, ws);
timesteps = cell(numel(reports) + 1, 1);
timesteps{1} = schedule.step.val;
for i = 1:numel(reports)
    [~, t] = convertReportToSchedule(reports{i}, schedule_small);
    timesteps{i+1} = t;
end
reps = vertcat({report}, reports);
iterations = zeros(numel(reps), 1);
for i = 1:numel(reps)
    r = reps{i};
    for j = 1:numel(reps{i})
        for k = 1:numel(r.ControlstepReports)
            if r.Converged
                iterations(i) = iterations(i) + r.ControlstepReports{k}.Iterations;
            end
        end
    end
end
small_val = schedule_small.step.val;
small_control = schedule_small.step.control;
bhp = cell(numel(wsols), 1);
qGs = cell(numel(wsols), 1);
for i = 1:numel(wsols)
    [~, ~, qGs{i}, bhp{i}] = wellSolToVector(wsols{i});
end
save(fullfile(here, 'ref_timestep_control_demo.mat'), 'timesteps', 'iterations', ...
     'small_val', 'small_control', 'bhp', 'qGs', '-v7');
