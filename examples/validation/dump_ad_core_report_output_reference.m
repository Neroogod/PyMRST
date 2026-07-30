%% MRST reference for getReportOutput.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

linear1 = struct('Iterations', [3], 'SolverTime', 2, 'PostProcessTime', 0.5);
linear2 = struct('Iterations', [9], 'SolverTime', 8, 'PostProcessTime', 0);
linear3 = struct('Iterations', [4], 'SolverTime', 1, 'PostProcessTime', 0.25);
linear4 = struct('Iterations', [8], 'SolverTime', 7, 'PostProcessTime', 0);
linear5 = struct('Iterations', [2], 'SolverTime', 0.75, 'PostProcessTime', 0.1);
n1 = struct('Solved', true, 'AssemblyTime', 1, 'LinearSolver', linear1);
n2 = struct('Solved', false, 'AssemblyTime', 0.2, 'LinearSolver', linear2);
n3 = struct('Solved', true, 'AssemblyTime', 0.5, 'LinearSolver', linear3);
n4 = struct('Solved', false, 'AssemblyTime', 0.1, 'LinearSolver', linear4);
n5 = struct('Solved', true, 'AssemblyTime', 0.25, 'LinearSolver', linear5);
failed = struct( ...
    'Converged', false, 'LocalTime', 4, ...
    'NonlinearReport', {{n1; n2}});
success = struct( ...
    'Converged', true, 'LocalTime', 2, ...
    'NonlinearReport', {{n3; n4}});
second = struct( ...
    'Converged', true, 'LocalTime', 3, ...
    'NonlinearReport', {{n5}});
control1 = struct('StepReports', {{failed; success}});
control2 = struct('StepReports', {{second}});
report = struct('ControlstepReports', {{control1; control2}});

default = getReportOutput(report);
linear = getReportOutput(report, 'type', 'linearIterations');
solver_time = getReportOutput( ...
    report, 'type', 'nonlinearSolverTime', 'ministeps', true);
linear_time = getReportOutput(report, 'type', 'linearSolverTime');

default_total = default.total;
default_wasted = default.wasted;
default_cuts = default.cuts;
default_time = default.time;
linear_total = linear.total;
linear_wasted = linear.wasted;
mini_total = solver_time.total;
mini_wasted = solver_time.wasted;
mini_cuts = solver_time.cuts;
mini_time = solver_time.time;
linear_time_total = linear_time.total;

save(fullfile(thisDir, 'ref_ad_core_report_output.mat'), '-v7', ...
    'default_total', 'default_wasted', 'default_cuts', 'default_time', ...
    'linear_total', 'linear_wasted', 'mini_total', 'mini_wasted', ...
    'mini_cuts', 'mini_time', 'linear_time_total');
fprintf('Wrote ref_ad_core_report_output.mat\n');
