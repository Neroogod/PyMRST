%% MRST reference for dependency, value-assignment and simulation-time helpers.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

x_double = assignValue([1; 2; 3], 9, 2);
x_ad = initVariablesADI([10; 20; 30]);
x_ad = assignValue(x_ad, [-2; -3], [1; 3]);
x_ad_value = x_ad.val;
x_ad_jacobian = full(x_ad.jac{1});

prop = struct();
prop.dependencies = {};
prop.externals = struct('name', {}, 'grouping', {});
prop = addPropertyDependence(prop, {'Pressure', 'Saturation'});
prop = addPropertyDependence(prop, 'Mobility', 'FlowPropertyFunctions');
supplied = struct('name', 'temperature', 'grouping', 'state');
prop = addPropertyDependence(prop, supplied);
dependencies = prop.dependencies;
external_names = {prop.externals.name};
external_groupings = {prop.externals.grouping};

control_report = struct('SimulationTime', 2);
control_times = getSimulationTime(cell(1, 1), control_report);

failed = struct('Converged', false, 'Timestep', 4);
ok1 = struct('Converged', true, 'Timestep', 1.5);
ok2 = struct('Converged', true, 'Timestep', 2.5);
control1 = struct('StepReports', {{failed, ok1}});
control2 = struct('StepReports', {{ok2}});
detailed = struct('ControlstepReports', {{control1, control2}});
ministep_times = getSimulationTime(cell(3, 1), detailed);

save(fullfile(thisDir, 'ref_ad_core_dependency_value_time.mat'), '-v7', ...
    'x_double', 'x_ad_value', 'x_ad_jacobian', 'dependencies', ...
    'external_names', 'external_groupings', 'control_times', 'ministep_times');
fprintf('Wrote ref_ad_core_dependency_value_time.mat\n');
