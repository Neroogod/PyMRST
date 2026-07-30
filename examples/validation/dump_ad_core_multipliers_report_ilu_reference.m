%% MRST reference for multipliers, report conversion, and ILU reordering.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

fluid = struct();
fluid.pvMultR = @(p) 1 + 0.01*p;
fluid.tranMultR = @(p) 2 + 0.02*p;
fluid.transMult = @(p) 3 + 0.03*p;
p = [10; 20]; p0 = [5; 15];
[pv, trans, mob, pv0, trans0, mob0] = getMultipliers(fluid, p, p0);

failed = struct('Converged', false, 'Timestep', 6);
ok1 = struct('Converged', true, 'Timestep', 4);
ok2 = struct('Converged', true, 'Timestep', 6);
ok3 = struct('Converged', true, 'Timestep', 20);
control1 = struct('StepReports', {{failed; ok1; ok2}});
control2 = struct('StepReports', {{ok3}});
report = struct('ControlstepReports', {{control1; control2}});
schedule = struct();
schedule.step = struct('val', [10; 20], 'control', [1; 2]);
schedule.control = {struct('name', 'A'), struct('name', 'B')};
[converted, timesteps, numsteps] = convertReportToSchedule(report, schedule);
converted_values = converted.step.val;
converted_controls = converted.step.control;

A = [4, 0, 0; 0, 0, 1; 0, 1, 2];
b = [10; 20; 30];
[A_reordered, b_reordered] = reorderForILU(A, b, 1);
[A_sparse, b_sparse] = reorderForILU(sparse(A), b, 1);
A_sparse = full(A_sparse);

save(fullfile(thisDir, 'ref_ad_core_multipliers_report_ilu.mat'), '-v7', ...
    'pv', 'trans', 'mob', 'pv0', 'trans0', 'mob0', ...
    'converted_values', 'converted_controls', 'timesteps', 'numsteps', ...
    'A_reordered', 'b_reordered', 'A_sparse', 'b_sparse');
fprintf('Wrote ref_ad_core_multipliers_report_ilu.mat\n');
