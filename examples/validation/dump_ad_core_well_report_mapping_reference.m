%% MRST reference for perforation mapping, incomp conversion and ministeps.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));
W = repmat(struct('name', '', 'cells', [], 'compi', []), 2, 1);
W(1).name = 'A'; W(1).cells = [1; 2]; W(1).compi = [1, 0];
W(2).name = 'B'; W(2).cells = 3; W(2).compi = [0.25, 0.75];
[perf2well, Rw] = getPerforationToWellMapping(W);
Rw_full = full(Rw);

old = repmat(struct('flux', [], 'pressure', 0), 2, 1);
old(1).flux = [-2; 3]; old(1).pressure = 100;
old(2).flux = -4; old(2).pressure = 90;
state = struct('wellSol', old);
fluid = struct();
fluid.properties = @(state) [1, 2];
fluid.saturation = @(state) zeros(3, 2);
fluid.relperm = @(s, state) [0.5, 0.5; 0.2, 0.8; 0.6, 0.4];
wellSols = convertIncompWellSols(W, state, fluid);
converted = wellSols{1};
names = {converted.name}; qWs = [converted.qWs];
qOs = [converted.qOs]; bhp = [converted.bhp];

failed = struct('Converged', false, 'Timestep', 4);
ok1 = struct('Converged', true, 'Timestep', 2);
ok2 = struct('Converged', true, 'Timestep', 3);
control1 = struct('StepReports', {{failed, ok1}});
control2 = struct('StepReports', {{ok2}});
report = struct('ControlstepReports', {{control1, control2}});
timesteps = getReportMinisteps(report);

save(fullfile(thisDir, 'ref_ad_core_well_report_mapping.mat'), '-v7', ...
    'perf2well', 'Rw_full', 'names', 'qWs', 'qOs', 'bhp', 'timesteps');
fprintf('Wrote ref_ad_core_well_report_mapping.mat\n');
