% Reference for getReportTimings and surfWithOutline
% (validate_ad_blackoil_example_utils.py): the timings of a fixed report,
% with and without skipConverged/total, and the surface and outline that
% surfWithOutline draws.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil
addpath(fullfile(ROOTDIR, 'modules', 'ad-blackoil', 'examples', 'utils'));
ls = struct('PreparationTime', 0.5, 'PostProcessTime', 0.25, 'LinearSolutionTime', 2.0);
nl = {struct('Converged', false, 'AssemblyTime', 1.0, 'LinearSolver', ls), ...
      struct('Converged', false, 'AssemblyTime', 1.5, 'LinearSolver', struct('SolverTime', 3.0)), ...
      struct('Converged', true, 'AssemblyTime', 0.75, 'LinearSolver', struct())};
step = struct('WallTime', 10.0, 'Iterations', 2);
step.StepReports = {struct('NonlinearReport', {nl})};
step2 = step; step2.WallTime = 20.0;
report = struct('ControlstepReports', {{step; step2}});
pack = @(t) [[t.Total]', [t.Assembly]', [t.LinearSolve]', [t.LinearSolvePrep]', ...
             [t.NumberOfAssemblies]', [t.Iterations]'];
timings = pack(getReportTimings(report));
timings_skip = pack(getReportTimings(report, 'skipConverged', true));
timings_total = pack(getReportTimings(report, 'total', true));
timings_cell = pack(getReportTimings(report.ControlstepReports));
[X, Y] = meshgrid(1:4, 1:3);
Z = X.*Y + 0.5*X;
f = figure('Visible', 'off');
surfWithOutline(X, Y, Z);
ax = gca;
s = findobj(ax, 'Type', 'surface');
surface_z = s.ZData;
lines = flipud(findobj(ax, 'Type', 'line'));   % creation order
outline = {};
for i = 1:numel(lines)
    outline{end+1} = [lines(i).XData(:), lines(i).YData(:), lines(i).ZData(:)]; %#ok
end
outline = vertcat(outline{:});
nlines = numel(lines);
color = lines(1).Color; width = lines(1).LineWidth;
held = ishold(ax);
close(f);
save(fullfile(here, 'ref_ad_blackoil_example_utils.mat'), '-v7', 'timings', 'timings_skip', ...
     'timings_total', 'timings_cell', 'X', 'Y', 'Z', 'surface_z', 'outline', 'nlines', ...
     'color', 'width', 'held');
