function dump_uplift_objective_reference(output_file)
if nargin < 1
    here = fileparts(mfilename('fullpath'));
    output_file = fullfile(here, 'ref_uplift_objectives.mat');
end
mrstModule add ad-core ad-mechanics optimization
model = AdMechanicsObjectiveFixtureModel();
state1 = struct('pressure', [1; 2], 'sW', [0.2; 0.3], ...
    'wellSol', struct([]), 'xd', [2; 3; 4]);
state2 = state1; state2.xd = [-3; 5; 6];
topnode = 2;
uplift = computeUpliftForState(model, state1, topnode);
uplift_ad = computeUpliftForState(model, state1, topnode, 'ComputePartials', true);
uplift_jac = full(uplift_ad.jac{end});
schedule.step.val = [4; 5];
linear = objUplift(model, {state1; state2}, schedule, topnode);
power = objUpliftAver(model, {state1; state2}, schedule, topnode, ...
    'exponent', 2, 'normalizationConstant', 0.5);
linear_values = cellfun(@double, linear);
power_values = cellfun(@double, power);
save(output_file, 'uplift', 'uplift_jac', 'linear_values', ...
    'power_values', 'schedule', 'state1', 'state2', 'topnode', '-v7');
fprintf('Wrote %s\n', output_file);
end
